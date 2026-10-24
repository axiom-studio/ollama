package launch

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/envconfig"
)

const (
	openSealReleasesURL = "https://github.com/axiom-studio/openseal/releases"
	// openSealMinVersion is the first release whose daemon can drive Ollama:
	// it selects schema-constrained structured output and a configurable
	// model timeout, and its TUI reads the API token from the environment.
	openSealMinVersion = "v1.0.20"
	// openSealModelTimeout bounds one model request. Local models can take
	// minutes for a single agent turn, well past OpenSeal's 90s default.
	openSealModelTimeout = "10m"
)

var (
	openSealReadyTimeout = 30 * time.Second
	openSealStopTimeout  = 10 * time.Second
)

// OpenSeal implements Runner for OpenSeal. It runs the OpenSeal daemon
// against Ollama for the lifetime of the OpenSeal terminal UI.
type OpenSeal struct{}

func (o *OpenSeal) String() string { return "OpenSeal" }

func (o *OpenSeal) Run(model string, _ []LaunchModel, args []string) error {
	bin, err := exec.LookPath("openseal")
	if err != nil {
		return fmt.Errorf("openseal is not installed, download it from %s", openSealReleasesURL)
	}
	if err := checkOpenSealVersion(bin); err != nil {
		return err
	}
	dir, err := openSealDataDir()
	if err != nil {
		return err
	}
	token, err := openSealToken()
	if err != nil {
		return err
	}

	// The TUI reads Ctrl+C as a key and exits by itself. These signals come
	// from outside the terminal, so forward them to the TUI and still stop
	// the daemon once it returns.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	daemon, err := startOpenSealDaemon(bin, dir, model, token)
	if err != nil {
		return err
	}
	defer daemon.stop()

	tui := exec.CommandContext(ctx, bin, append([]string{"tui", "--endpoint", daemon.endpoint}, args...)...)
	if runtime.GOOS != "windows" {
		tui.Cancel = func() error { return tui.Process.Signal(os.Interrupt) }
		tui.WaitDelay = openSealStopTimeout
	}
	tui.Stdin = os.Stdin
	tui.Stdout = os.Stdout
	tui.Stderr = os.Stderr
	tui.Env = append(os.Environ(), "OPENSEAL_API_TOKEN="+token)
	return tui.Run()
}

func checkOpenSealVersion(bin string) error {
	out, err := exec.Command(bin, "version").Output()
	if err != nil {
		return fmt.Errorf("failed to get openseal version from %s: %w", bin, err)
	}
	// Output is "OpenSeal v<semver>".
	version := strings.TrimSpace(string(out))
	if fields := strings.Fields(version); len(fields) > 0 {
		version = fields[len(fields)-1]
	}
	if !semver.IsValid(version) {
		return fmt.Errorf("unexpected openseal version output: %q", strings.TrimSpace(string(out)))
	}
	if semver.Compare(version, openSealMinVersion) < 0 {
		return fmt.Errorf("openseal version %s is too old, minimum required is %s, download a newer release from %s", version, openSealMinVersion, openSealReleasesURL)
	}
	return nil
}

func openSealDataDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "launch", "openseal"), nil
}

// openSealToken returns a fresh API token for one daemon. The daemon requires
// a token in desktop operator mode, and the TUI is its only client.
func openSealToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("failed to generate openseal API token: %w", err)
	}
	return hex.EncodeToString(b), nil
}

func openSealBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/") + "/v1"
}

func openSealDaemonArgs(dir string) []string {
	return []string{
		"daemon",
		"--config", filepath.Join(dir, "daemon.yaml"),
		"--context", filepath.Join(dir, "context.yaml"),
		"--listen", "127.0.0.1:0",
		"--standalone-operator",
		// Agent turns use the model only in desktop operator mode.
		"--desktop-operator",
	}
}

func openSealDaemonEnv(model, token string) []string {
	return append(os.Environ(),
		"OPENSEAL_API_TOKEN="+token,
		"OPENSEAL_LLM_BASE_URL="+openSealBaseURL(),
		"OPENSEAL_LLM_MODEL="+model,
		"OPENAI_API_KEY=ollama",
		// Ollama ignores tool_choice, so a forced tool call is not reliable.
		// A JSON schema response format constrains the output instead.
		"OPENSEAL_LLM_STRUCTURED_OUTPUT=json_schema",
		"OPENSEAL_LLM_TIMEOUT="+openSealModelTimeout,
	)
}

type openSealDaemon struct {
	cmd      *exec.Cmd
	done     chan struct{}
	endpoint string
}

// startOpenSealDaemon starts the daemon and waits for its readiness line,
// "OPENSEAL_DAEMON {"type":"ready","endpoint":"http://127.0.0.1:<port>"}".
func startOpenSealDaemon(bin, dir, model, token string) (*openSealDaemon, error) {
	logDir := filepath.Join(dir, "logs")
	if err := os.MkdirAll(logDir, 0o700); err != nil {
		return nil, fmt.Errorf("failed to create openseal data directory %s: %w", logDir, err)
	}
	logPath := filepath.Join(logDir, "daemon.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("failed to open openseal daemon log %s: %w", logPath, err)
	}
	defer logFile.Close()

	// A plain pipe rather than StdoutPipe: Wait then does not depend on the
	// pipe closing, which a process the daemon starts could hold open.
	stdout, stdoutWriter, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create openseal daemon output pipe: %w", err)
	}
	cmd := exec.Command(bin, openSealDaemonArgs(dir)...)
	cmd.Dir = dir
	cmd.Env = openSealDaemonEnv(model, token)
	cmd.Stdout = stdoutWriter
	cmd.Stderr = logFile
	err = cmd.Start()
	stdoutWriter.Close()
	if err != nil {
		stdout.Close()
		return nil, fmt.Errorf("failed to start openseal daemon: %w", err)
	}

	d := &openSealDaemon{cmd: cmd, done: make(chan struct{})}
	ready := make(chan string, 1)
	go func() {
		defer stdout.Close()
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			if endpoint, ok := parseOpenSealReady(scanner.Text()); ok {
				ready <- endpoint
				break
			}
		}
		// Keep draining so the daemon never blocks on a full pipe.
		_, _ = io.Copy(io.Discard, stdout)
	}()
	go func() {
		_ = cmd.Wait()
		close(d.done)
	}()

	select {
	case d.endpoint = <-ready:
		return d, nil
	case <-d.done:
		return nil, fmt.Errorf("openseal daemon exited before it was ready, see %s", logPath)
	case <-time.After(openSealReadyTimeout):
		d.stop()
		return nil, fmt.Errorf("openseal daemon was not ready after %s, see %s", openSealReadyTimeout, logPath)
	}
}

func parseOpenSealReady(line string) (string, bool) {
	payload, ok := strings.CutPrefix(line, "OPENSEAL_DAEMON ")
	if !ok {
		return "", false
	}
	var msg struct {
		Type     string `json:"type"`
		Endpoint string `json:"endpoint"`
	}
	if err := json.Unmarshal([]byte(payload), &msg); err != nil || msg.Type != "ready" || msg.Endpoint == "" {
		return "", false
	}
	return msg.Endpoint, true
}

// stop asks the daemon to shut down and kills it if it does not exit in time.
// Windows cannot send an interrupt to a child process, so it is killed there.
func (d *openSealDaemon) stop() {
	if d.cmd.Process.Signal(os.Interrupt) != nil {
		_ = d.cmd.Process.Kill()
	}
	select {
	case <-d.done:
	case <-time.After(openSealStopTimeout):
		_ = d.cmd.Process.Kill()
		<-d.done
	}
}
