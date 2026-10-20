# Regenerate docs/static/demo.gif: `just demo` (needs vhs, tmux, uv, claude and the scw CLI)
Output docs/static/demo.gif
Require tmux
Require claude
Require scw
Set Shell bash
Set FontSize 16
Set Width 1800
Set Height 900
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# Throwaway HOME and a working-tree copy under it, so no real home path shows in the banner
Hide
Type `export PADWAN_API_KEY=$(scw config get secret-key) UV_CACHE_DIR=$(uv cache dir) DEMO_HOME=/tmp/demo; rm -rf "$DEMO_HOME"; mkdir -p "$DEMO_HOME" && rsync -a --exclude .venv --exclude .git --exclude .scratch --exclude .env "$PWD/" "$DEMO_HOME/padwan-proxy/" && cd "$DEMO_HOME/padwan-proxy" && uv sync -q; export HOME=$DEMO_HOME`
Enter
Type `unset CLAUDECODE $(compgen -v CLAUDE_CODE_); mkdir ~/.claude-padwan`
Enter
Type `echo '{"hasCompletedOnboarding":true,"projects":{"'$PWD'":{"hasTrustDialogAccepted":true}}}' > ~/.claude-padwan/.claude.json; echo setup""-ok`
Enter
Wait+Screen@300s /setup-ok/
# Left pane: Claude Code. Right pane: the proxy and its request tree.
Type `fuser -k 4010/tcp 2>/dev/null; tmux -L demo kill-server 2>/dev/null; tmux -L demo -f /dev/null new -d -s demo -x 200 -y 50 'env PS1="$ " bash --norc' \; split-window -h -l 50% 'env PS1="$ " bash --norc' \; set -g status off && tmux -L demo attach -t demo`
Enter
Sleep 1s
Show

Copy "uv run padwan-proxy --backend-url https://api.scaleway.ai/v1/ -m glm-5.2 --small-model qwen3.6-35b-a3b --port 4010 --context-window 256000 --claude-config ~/.claude-padwan -v --rich --breakdown"
Paste
Sleep 500ms
Enter
Sleep 3s
Ctrl+B
Left
Type "CLAUDE_CONFIG_DIR=~/.claude-padwan claude"
Enter
Sleep 8s
Copy "In two sentences, what does padwan_proxy/claude_config.py do?"
Paste
Sleep 500ms
Enter
Sleep 20s
Ctrl+C
Ctrl+C
Sleep 1s

Hide
Type `tmux -L demo kill-server; cd /; rm -rf "$DEMO_HOME"`
Enter
Sleep 1s
