#!/usr/bin/env bash

# Usage:
#   ./add_progressbar.sh input.mp4 output.mp4 [bar_height]
#
# If bar_height is not given, defaults to 10.

input="$1"
output="$2"
bar_height="${3:-10}"  # default is 10 if not provided

# 1) Extract duration, width, and height from input
duration=$(ffprobe -v error -show_entries format=duration \
           -of csv=p=0 "$input")
width=$(ffprobe -v error -select_streams v:0 \
         -show_entries stream=width -of csv=p=0 "$input")
height=$(ffprobe -v error -select_streams v:0 \
          -show_entries stream=height -of csv=p=0 "$input")

# 2) Run ffmpeg
#
# Breakdown of filter_complex steps:
#   1) pad=...        => Increase canvas height by bar_height (adds empty space below).
#   2) color=...      => Create a red strip sized [width x bar_height].
#   3) overlay=...    => Animate that strip across the bottom region.
#
# The expression x='-W + (W/duration)*t' moves the bar from left to right.
# The y positioning is set to the bottom of the original content (y=height).
#
# For clarity:
#   - "W" in overlay is the *padded* output width, which is the same as the original width.
#   - "t" is the timestamp in seconds.
#   - shortest=1 ensures we stop overlaying once the video ends.

ffmpeg -i "$input" \
  -filter_complex "
    [0:v]pad=width=${width}:height=${height}+${bar_height}:x=0:y=0:color=black[padded];
    color=c=0x4c72b0:s=${width}x${bar_height}[bar];
    [padded][bar]overlay=
      x='-W + (W/${duration})*t':
      y=${height}:
      shortest=1
  " \
  -c:a copy \
  "$output"
