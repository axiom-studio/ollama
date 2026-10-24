package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fakeOpenSealScript stands in for the openseal binary. It records what it
// was started with in $FAKE_LOG; FAKE_VERSION, FAKE_DAEMON and FAKE_TUI_EXIT
// select its behaviour.
const fakeOpenSealScript = `#!/bin/sh
log="$FAKE_LOG"
case "$1" in
version)
	echo "OpenSeal ${FAKE_VERSION:-v1.0.20}"
	;;
daemon)
	printf 'daemon_args=%s\n' "$*" >> "$log"
	printf 'daemon_dir=%s\n' "$(pwd)" >> "$log"
	printf 'daemon_env=%s|%s|%s|%s|%s\n' "$OPENSEAL_LLM_BASE_URL" "$OPENSEAL_LLM_MODEL" "$OPENAI_API_KEY" "$OPENSEAL_LLM_STRUCTURED_OUTPUT" "$OPENSEAL_LLM_TIMEOUT" >> "$log"
	printf 'daemon_token=%s\n' "$OPENSEAL_API_TOKEN" >> "$log"
	echo "daemon stderr line" >&2
	case "$FAKE_DAEMON" in
	exit) exit 1 ;;
	hang) ;;
	*)
		echo "unrelated output"
		echo 'OPENSEAL_DAEMON {"type":"ready","endpoint":"http://127.0.0.1:43210"}'
		;;
	esac
	if [ "$FAKE_DAEMON" = ignore ]; then
		trap '' INT
	else
		trap 'echo daemon_stopped=INT >> "$log"; exit 0' INT
	fi
	while :; do sleep 0.05; done
	;;
tui)
	printf 'tui_args=%s\n' "$*" >> "$log"
	printf 'tui_token=%s\n' "$OPENSEAL_API_TOKEN" >> "$log"
	exit "${FAKE_TUI_EXIT:-0}"
	;;
esac
`

type fakeOpenSeal struct {
	home    string
	logPath string
}

func setupFakeOpenSeal(t *testing.T) fakeOpenSeal {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	home := t.TempDir()
	setLaunchTestHome(t, home)
	binDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(binDir, "openseal"), []byte(fakeOpenSealScript), 0o755); err != nil {
		t.Fatalf("failed to write fake openseal binary: %v", err)
	}
	// The fake needs sh, printf and sleep from the system PATH.
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	f := fakeOpenSeal{home: home, logPath: filepath.Join(t.TempDir(), "openseal.log")}
	t.Setenv("FAKE_LOG", f.logPath)
	return f
}

// values returns every value recorded for key, in order.
func (f fakeOpenSeal) values(t *testing.T, key string) []string {
	t.Helper()
	data, err := os.ReadFile(f.logPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatalf("failed to read fake openseal log: %v", err)
	}
	var values []string
	for _, line := range strings.Split(string(data), "\n") {
		if value, ok := strings.CutPrefix(line, key+"="); ok {
			values = append(values, value)
		}
	}
	return values
}

func (f fakeOpenSeal) value(t *testing.T, key string) string {
	t.Helper()
	values := f.values(t, key)
	if len(values) != 1 {
		t.Fatalf("expected one %s entry in fake openseal log, got %q", key, values)
	}
	return values[0]
}

func TestOpenSealRunConnectsTUIToDaemonOnOllama(t *testing.T) {
	f := setupFakeOpenSeal(t)

	if err := (&OpenSeal{}).Run("qwen2.5:7b", nil, []string{"--poll", "1s"}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	dir := filepath.Join(f.home, ".ollama", "launch", "openseal")
	wantArgs := "daemon --config " + filepath.Join(dir, "daemon.yaml") +
		" --context " + filepath.Join(dir, "context.yaml") +
		" --listen 127.0.0.1:0 --standalone-operator --desktop-operator"
	if got := f.value(t, "daemon_args"); got != wantArgs {
		t.Fatalf("daemon args = %q, want %q", got, wantArgs)
	}
	if got := f.value(t, "daemon_dir"); got != dir {
		t.Fatalf("daemon working directory = %q, want %q", got, dir)
	}
	wantEnv := "http://127.0.0.1:11434/v1|qwen2.5:7b|ollama|json_schema|10m"
	if got := f.value(t, "daemon_env"); got != wantEnv {
		t.Fatalf("daemon env = %q, want %q", got, wantEnv)
	}
	if got := f.value(t, "tui_args"); got != "tui --endpoint http://127.0.0.1:43210 --poll 1s" {
		t.Fatalf("tui args = %q, want the daemon endpoint followed by extra args", got)
	}

	daemonToken, tuiToken := f.value(t, "daemon_token"), f.value(t, "tui_token")
	if len(daemonToken) != 64 {
		t.Fatalf("daemon token = %q, want a 64 character hex token", daemonToken)
	}
	if tuiToken != daemonToken {
		t.Fatalf("tui token %q does not match daemon token %q", tuiToken, daemonToken)
	}
	for _, key := range []string{"daemon_args", "tui_args"} {
		if strings.Contains(f.value(t, key), daemonToken) {
			t.Fatalf("API token leaked into %s", key)
		}
	}

	if got := f.values(t, "daemon_stopped"); len(got) != 1 || got[0] != "INT" {
		t.Fatalf("expected the daemon to be interrupted once after the TUI exited, got %q", got)
	}
	logData, err := os.ReadFile(filepath.Join(dir, "logs", "daemon.log"))
	if err != nil {
		t.Fatalf("failed to read daemon log: %v", err)
	}
	if !strings.Contains(string(logData), "daemon stderr line") {
		t.Fatalf("expected daemon stderr in daemon.log, got %q", logData)
	}
}

func TestOpenSealRunUsesFreshTokenPerLaunch(t *testing.T) {
	f := setupFakeOpenSeal(t)

	for range 2 {
		if err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil); err != nil {
			t.Fatalf("Run returned error: %v", err)
		}
	}
	tokens := f.values(t, "daemon_token")
	if len(tokens) != 2 || tokens[0] == tokens[1] {
		t.Fatalf("expected two different tokens for two launches, got %q", tokens)
	}
}

func TestOpenSealRunReturnsTUIErrorAndStopsDaemon(t *testing.T) {
	f := setupFakeOpenSeal(t)
	t.Setenv("FAKE_TUI_EXIT", "3")

	err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "exit status 3") {
		t.Fatalf("expected the TUI exit status as error, got %v", err)
	}
	if got := f.values(t, "daemon_stopped"); len(got) != 1 {
		t.Fatalf("expected the daemon to be stopped after the TUI failed, got %q", got)
	}
}

func TestOpenSealRunKillsDaemonThatIgnoresInterrupt(t *testing.T) {
	f := setupFakeOpenSeal(t)
	t.Setenv("FAKE_DAEMON", "ignore")
	prev := openSealStopTimeout
	openSealStopTimeout = 200 * time.Millisecond
	t.Cleanup(func() { openSealStopTimeout = prev })

	start := time.Now()
	if err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("Run took %s, expected the daemon to be killed after the stop timeout", elapsed)
	}
	if got := f.values(t, "tui_args"); len(got) != 1 {
		t.Fatalf("expected the TUI to run once, got %q", got)
	}
}

func TestOpenSealRunReportsDaemonThatExitsBeforeReady(t *testing.T) {
	f := setupFakeOpenSeal(t)
	t.Setenv("FAKE_DAEMON", "exit")

	err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil)
	logPath := filepath.Join(f.home, ".ollama", "launch", "openseal", "logs", "daemon.log")
	if err == nil || !strings.Contains(err.Error(), "exited before it was ready") || !strings.Contains(err.Error(), logPath) {
		t.Fatalf("expected an early exit error naming %s, got %v", logPath, err)
	}
	if got := f.values(t, "tui_args"); len(got) != 0 {
		t.Fatalf("expected the TUI not to run, got %q", got)
	}
}

func TestOpenSealRunStopsDaemonThatNeverBecomesReady(t *testing.T) {
	f := setupFakeOpenSeal(t)
	t.Setenv("FAKE_DAEMON", "hang")
	prev := openSealReadyTimeout
	openSealReadyTimeout = 200 * time.Millisecond
	t.Cleanup(func() { openSealReadyTimeout = prev })

	err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "was not ready after 200ms") {
		t.Fatalf("expected a readiness timeout error, got %v", err)
	}
	if got := f.values(t, "daemon_stopped"); len(got) != 1 {
		t.Fatalf("expected the daemon to be stopped after the readiness timeout, got %q", got)
	}
	if got := f.values(t, "tui_args"); len(got) != 0 {
		t.Fatalf("expected the TUI not to run, got %q", got)
	}
}

func TestOpenSealRunRequiresMinimumVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
		wantErr string
	}{
		{name: "older release", version: "v1.0.19", wantErr: "openseal version v1.0.19 is too old, minimum required is v1.0.20"},
		{name: "prerelease of minimum", version: "v1.0.20-dev.4", wantErr: "is too old"},
		{name: "unstamped build", version: "v0.0.0-dev", wantErr: "is too old"},
		{name: "unparseable output", version: "unknown", wantErr: `unexpected openseal version output: "OpenSeal unknown"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := setupFakeOpenSeal(t)
			t.Setenv("FAKE_VERSION", tt.version)

			err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
			if tt.name != "unparseable output" && !strings.Contains(err.Error(), openSealReleasesURL) {
				t.Fatalf("expected the releases URL in %q", err)
			}
			if got := f.values(t, "daemon_args"); len(got) != 0 {
				t.Fatalf("expected no daemon to start, got %q", got)
			}
		})
	}
}

func TestOpenSealRunAcceptsNewerVersion(t *testing.T) {
	f := setupFakeOpenSeal(t)
	t.Setenv("FAKE_VERSION", "v1.2.0")

	if err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if got := f.values(t, "tui_args"); len(got) != 1 {
		t.Fatalf("expected the TUI to run once, got %q", got)
	}
}

func TestOpenSealRunNotInstalled(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	err := (&OpenSeal{}).Run("qwen2.5:7b", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "openseal is not installed") || !strings.Contains(err.Error(), openSealReleasesURL) {
		t.Fatalf("expected a not installed error with the releases URL, got %v", err)
	}
}

func TestParseOpenSealReady(t *testing.T) {
	tests := []struct {
		name         string
		line         string
		wantEndpoint string
		wantOK       bool
	}{
		{name: "ready", line: `OPENSEAL_DAEMON {"type":"ready","endpoint":"http://127.0.0.1:8123"}`, wantEndpoint: "http://127.0.0.1:8123", wantOK: true},
		{name: "other output", line: "starting kernel"},
		{name: "empty line", line: ""},
		{name: "other message type", line: `OPENSEAL_DAEMON {"type":"error","endpoint":"http://127.0.0.1:8123"}`},
		{name: "missing endpoint", line: `OPENSEAL_DAEMON {"type":"ready"}`},
		{name: "invalid json", line: `OPENSEAL_DAEMON {"type":`},
		{name: "prefix not at start", line: ` OPENSEAL_DAEMON {"type":"ready","endpoint":"http://127.0.0.1:8123"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			endpoint, ok := parseOpenSealReady(tt.line)
			if endpoint != tt.wantEndpoint || ok != tt.wantOK {
				t.Fatalf("parseOpenSealReady(%q) = %q, %v, want %q, %v", tt.line, endpoint, ok, tt.wantEndpoint, tt.wantOK)
			}
		})
	}
}
