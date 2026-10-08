# Regenerate docs/static/demo.gif: `just demo` (needs vhs, docker, claude and the compose .env)
Output docs/static/demo.gif
Require docker
Require claude
Set Shell bash
Set FontSize 16
Set Width 1200
Set Height 700
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# Throwaway HOME with Claude Code onboarding and folder trust pre-accepted
Hide
Type `export DEMO_HOME=$(mktemp -d) HOST_UID=$(id -u) HOST_GID=$(id -g) PS1='$ '; export HOME=$DEMO_HOME`
Enter
Type `unset CLAUDECODE $(compgen -v CLAUDE_CODE_); mkdir ~/.claude-padwan`
Enter
Type `echo '{"hasCompletedOnboarding":true,"projects":{"'$PWD'":{"hasTrustDialogAccepted":true}}}' > ~/.claude-padwan/.claude.json; clear`
Enter
Type `docker compose build -q && echo built""-ok`
Enter
Wait+Screen@600s /built-ok/
Type "clear"
Enter
Sleep 500ms
Show

Type "docker compose up -d"
Enter
Sleep 4s
Type "CLAUDE_CONFIG_DIR=~/.claude-padwan claude"
Enter
Sleep 8s
Type "In two sentences, what does padwan_proxy/claude_config.py do?"
Sleep 500ms
Enter
Sleep 15s
Ctrl+C
Ctrl+C
Sleep 1s

Hide
Type `docker compose down; rm -rf "$DEMO_HOME"`
Enter
Sleep 3s
