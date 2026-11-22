#!/usr/bin/env bash
# Log NVMe / CPU / GPU temperatures and fan speeds once a minute (CSV) to correlate throttling with jobs.
OUT=/home/alliedtoasters/projects/chess/bot/data/allie/grow/temps.csv
[ -f $OUT ] || echo "time,nvme_composite,nvme_s1,nvme_s2,cpu_tctl,gpu,gpu_util,cpu_fan,intake_fan,warden_job" > $OUT
H=/sys/class/hwmon
nv=$(grep -l nvme $H/*/name | xargs dirname); k=$(grep -l k10temp $H/*/name | xargs dirname); f=$(grep -l system76 $H/*/name | xargs dirname)
while true; do
  g=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  j=$(~/projects/clearwood/warden/.venv/bin/warden status 2>/dev/null | grep -o '"label": "[^"]*"' | head -1 | cut -d'"' -f4)
  echo "$(date +%FT%T),$(($(cat $nv/temp1_input)/1000)),$(($(cat $nv/temp2_input)/1000)),$(($(cat $nv/temp3_input)/1000)),$(($(cat $k/temp1_input)/1000)),$g,$(cat $f/fan1_input),$(cat $f/fan2_input),$j" >> $OUT
  sleep 60
done
