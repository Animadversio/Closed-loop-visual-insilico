#!/bin/bash
shopt -s globstar nullglob nocaseglob
for mp4 in /Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/VVS_Accentuation_project/Figures/red_20250428-30_accentuation/*/*.mp4; do
  # 1) get total duration in seconds (requires ffprobe)
  duration=$(ffprobe -v error \
           -select_streams v:0 \
           -show_entries format=duration \
           -of default=noprint_wrappers=1:nokey=1 \
           "$mp4")
  # 2) compute midpoint
  mid=$(awk "BEGIN { printf \"%.3f\", $duration/2 }")
  # 3) build output name
  out="${mp4%.*}_midframe.png"
  echo "Extracting mid-frame of $mp4 at ${mid}s → $out"
  # 4) seek then grab 1 frame
  ffmpeg -ss "$mid" -i "$mp4" -frames:v 1 -q:v 2 "$out"
done
