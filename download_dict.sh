#!/usr/bin/env sh
set -e

URL="https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
DEST="tmp/raw-wiktextract-data.jsonl"
DEST_GZ="tmp/raw-wiktextract-data.jsonl.gz"

mkdir -p tmp

if [ -f "$DEST" ]; then
    printf "Data file already exists. Re-download? [y/N]: "
    read redownload
    case "${redownload}" in
        y|Y) ;;
        *) echo "Skipping download."; exit 0 ;;
    esac
fi

echo "Downloading Wiktionary dump (2.5 GB compressed)..."
curl -fL --progress-bar "$URL" -o "$DEST_GZ"

echo "Decompressing (21 GB uncompressed)..."
gunzip -f "$DEST_GZ"

echo "Done. Data written to $DEST"
