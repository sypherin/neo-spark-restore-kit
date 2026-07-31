#!/usr/bin/env bash
# spark-stack-guard — keep the Neo stack alive on the DGX Spark.
# Runs from a systemd user timer every 2 minutes:
#   1. every unit enabled (survives reboot) — enables any that are not
#   2. every unit active — restarts any that died and stayed down
#   3. every port answers its health check — restarts the owning unit if not
# Logs one line per action to the journal (view: journalctl --user -t spark-guard).
# Deliberately boring: no LLM, no network calls off-box, nothing destructive.
set -u
log() { echo "$*" | systemd-cat -t spark-guard -p "${2:-info}"; }

# unit:port:health-path  (port empty = unit-only check)
CHECKS="llama-gemma:8001:/health llama-surya2:8093:/health llama-vlm:8080:/health surya:8090:/healthz llm-gateway:8002:/healthz"

for spec in $CHECKS; do
  unit="${spec%%:*}"; rest="${spec#*:}"; port="${rest%%:*}"; path="${rest#*:}"

  if [ "$(systemctl --user is-enabled "$unit" 2>/dev/null)" != "enabled" ]; then
    systemctl --user enable "$unit" >/dev/null 2>&1 \
      && log "enabled $unit (was not enabled — would not survive reboot)" warning
  fi

  state="$(systemctl --user is-active "$unit" 2>/dev/null)"
  if [ "$state" = "failed" ] || [ "$state" = "inactive" ]; then
    systemctl --user reset-failed "$unit" >/dev/null 2>&1
    systemctl --user restart "$unit" >/dev/null 2>&1 \
      && log "restarted $unit (was $state)" warning \
      || log "FAILED to restart $unit — journalctl --user -u $unit" err
    continue   # give it a cycle to come up before health-probing
  fi

  # activating = mid-start (model load can take minutes) — leave it alone
  [ "$state" = "activating" ] && continue

  if [ -n "$port" ]; then
    if ! curl -sf -m 10 "http://127.0.0.1:${port}${path}" >/dev/null 2>&1; then
      # active but not answering: give slow model loads grace — only restart
      # if the unit has been active for over 10 minutes and still won't answer
      started=$(systemctl --user show "$unit" -p ActiveEnterTimestampMonotonic --value 2>/dev/null || echo 0)
      now=$(awk '{printf "%d", $1*1000000}' /proc/uptime)
      if [ -n "$started" ] && [ "$started" -gt 0 ] && [ $((now - started)) -gt 600000000 ]; then
        systemctl --user restart "$unit" >/dev/null 2>&1 \
          && log "restarted $unit (active but :${port}${path} unresponsive >10min)" warning
      else
        log "$unit active, :${port}${path} not answering yet (warmup grace)" info
      fi
    fi
  fi
done

# linger: without it every user service dies at logout / doesn't start at boot
if ! loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
  loginctl enable-linger "$USER" 2>/dev/null \
    && log "enabled linger for $USER (user services now survive logout/boot)" warning
fi
