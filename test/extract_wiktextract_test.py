import json
import os

LEXICON_PATH = "../dic/lexicon.jsonl"
WIKTEXTRACT_PATH = "../tmp/raw-wiktextract-data.jsonl"
OUTPUT_PATH = "../tmp/wiktextract_test_data.jsonl"

def load_lexicon_phrases():
    phrases = set()
    with open(LEXICON_PATH, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            phrase = entry["phrase"].lower()
            if " " in phrase or "-" in phrase:
                phrases.add(phrase)
    return phrases

def extract_test_data(lexicon_phrases):
    print("Extracting test sentences from wiktextract...")

    if not os.path.exists(WIKTEXTRACT_PATH):
        print(f"Error: {WIKTEXTRACT_PATH} not found.")
        return

    count = 0
    with open(WIKTEXTRACT_PATH, 'r') as f_in, open(OUTPUT_PATH, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            word = entry.get("word", "").lower()

            # Only care about multi-word phrases that are in our lexicon
            if word in lexicon_phrases:
                for sense in entry.get("senses", []):
                    for example in sense.get("examples", []):
                        if "text" in example:
                            text = example["text"].strip()
                            # Basic length filtering to avoid absurdly long or short examples
                            if 15 < len(text) < 300:
                                json.dump({"text": text, "expected_phrase": word}, f_out)
                                f_out.write("\n")
                                count += 1

    print(f"Extracted and saved {count} test sentences to {OUTPUT_PATH}.")

if __name__ == "__main__":
    print("Loading Lexicon...")
    lexicon_phrases = load_lexicon_phrases()
    print(f"Loaded {len(lexicon_phrases)} multi-word phrases from lexicon.")
    extract_test_data(lexicon_phrases)
