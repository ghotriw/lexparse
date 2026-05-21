#!/usr/bin/env sh
set -e

echo "Select model:"
echo "  1) deberta-v3-xsmall-ewt-gum  (fastest, lowest RAM)"
echo "  2) deberta-v3-xsmall-ewt      (fastest, lowest RAM)"
echo "  3) deberta-v3-small-ewt       (faster, less RAM)"
echo "  4) deberta-v3-base-ewt        (more accurate)"
printf "Choice [1]: "
read choice

case "${choice}" in
    4) REPO="ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt" ;;
    3) REPO="ghotriw/deberta-v3-small-biaffine-dep-pos-en-ewt" ;;
    2) REPO="ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt" ;;
    *) REPO="ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt-gum" ;;
esac

BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p model dic

FILES="model/model.fp16.onnx model/vocabs.json model/tokenizer.json"

existing=0
total=0
for f in $FILES; do
    total=$((total + 1))
    [ -f "$f" ] && existing=$((existing + 1))
done

FORCE=0
if [ "$existing" = "$total" ]; then
    printf "All model files already exist. Re-download? [y/N]: "
    read redownload
    case "${redownload}" in y|Y) FORCE=1 ;; esac
elif [ "$existing" -gt 0 ]; then
    printf "Partial download detected ($existing/$total files). Re-download all? [y/N]: "
    read redownload
    case "${redownload}" in y|Y) FORCE=1 ;; esac
fi

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ] && [ "$FORCE" = "0" ]; then
        echo "skip $dest (already exists)"
        return
    fi
    echo "downloading $dest ..."
    curl -fL --progress-bar "$url" -o "$dest"
}

download "${BASE}/model.fp16.onnx"       model/model.fp16.onnx
download "${BASE}/vocabs.json"           model/vocabs.json
download "${BASE}/tokenizer.json"        model/tokenizer.json

echo "done"
