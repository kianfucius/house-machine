#!/bin/bash

# Set the input and output directories
input_dir="data/unprocessed_songs"
output_dir="data/unprocessed_songs_wav"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all FLAC files in the input directory
for flac_file in "$input_dir"/*.flac; do
  # Extract the file name without extension
  file_name=$(basename -- "$flac_file")
  file_name_no_ext="${file_name%.*}"

  # Set the output WAV file path
  wav_file="$output_dir/$file_name_no_ext.wav"

  # Use FFmpeg to convert FLAC to WAV
  ffmpeg -i "$flac_file" "$wav_file"
done
