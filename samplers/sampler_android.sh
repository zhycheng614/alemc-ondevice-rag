#!/data/data/com.termux/files/usr/bin/bash
# Energy sampler for the Snapdragon 8 Elite (Android / Termux) phone.
#
# Reads the fuel-gauge sysfs nodes at ~10 Hz and logs instantaneous power.
#   current_now : microamps  (uA);  sign convention varies by device
#   voltage_now : microvolts (uV)
#   P[W] = |current_now| * voltage_now / 1e12
#
# EXPERIMENT_PLAN.md §7.1. Run UNPLUGGED, screen off, airplane mode,
# background apps closed. Root preferred for /sys access; some devices expose
# these nodes without root.
#
# Usage:
#   # 60s idle baseline (phone idle, before the run):
#   ./sampler_android.sh results/raw/idle_phone.csv 60
#   # during the pipeline run (Ctrl-C to stop, or pass a duration):
#   ./sampler_android.sh results/raw/energy_phone.csv
#
# Output CSV columns: t_epoch, current_ua, voltage_uv
# Feed to samplers/energy_integrate.py alongside the run's per-query CSV.

set -u
OUT="${1:?usage: sampler_android.sh <out.csv> [duration_s]}"
DURATION="${2:-}"

# Locate the battery power-supply node (varies across SoCs/vendors).
BATT=""
for d in /sys/class/power_supply/battery \
         /sys/class/power_supply/bms \
         /sys/class/power_supply/qcom-battery; do
  if [ -r "$d/current_now" ] && [ -r "$d/voltage_now" ]; then
    BATT="$d"; break
  fi
done

if [ -z "$BATT" ]; then
  echo "[sampler_android] ERROR: no readable current_now/voltage_now node." >&2
  echo "  Try: su -c 'cat /sys/class/power_supply/battery/current_now'" >&2
  echo "  Fallback: adb shell dumpsys batterystats --charged (coarse)." >&2
  exit 2
fi
echo "[sampler_android] using $BATT -> $OUT" >&2

mkdir -p "$(dirname "$OUT")"
echo "t_epoch,current_ua,voltage_uv" > "$OUT"

END=""
if [ -n "$DURATION" ]; then
  END=$(( $(date +%s) + DURATION ))
fi

while true; do
  T=$(date +%s.%N)
  I=$(cat "$BATT/current_now" 2>/dev/null)
  V=$(cat "$BATT/voltage_now" 2>/dev/null)
  if [ -n "$I" ] && [ -n "$V" ]; then
    echo "$T,$I,$V" >> "$OUT"
  fi
  if [ -n "$END" ] && [ "$(date +%s)" -ge "$END" ]; then
    break
  fi
  sleep 0.1
done
echo "[sampler_android] done -> $OUT" >&2
