#!/usr/bin/env sh
set -e

echo "Select model:"
echo "  1) deberta-v3-small  (faster, less RAM)"
echo "  2) deberta-v3-base   (more accurate)"
printf "Choice [1]: "
read choice

case "${choice}" in
    2) REPO="ghotriw/deberta-v3-base-biaffine-dep-pos-en" ;;
    *) REPO="ghotriw/deberta-v3-small-biaffine-dep-pos-en" ;;
esac

BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p model dic

FILES="model/model.fp16.onnx model/vocabs.json model/idiom_classifier.json model/tokenizer.json dic/lexicon.json dic/phrasal-verbs.json"

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
download "${BASE}/idiom_classifier.json" model/idiom_classifier.json
download "${BASE}/tokenizer.json"        model/tokenizer.json
download "${BASE}/lexicon.json"          dic/lexicon.json
download "${BASE}/phrasal-verbs.json"    dic/phrasal-verbs.json

echo "done"
