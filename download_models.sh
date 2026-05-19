#!/usr/bin/env sh
set -e

REPO="ghotriw/deberta-v3-small-biaffine-dep-pos-en"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p model dic

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
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
