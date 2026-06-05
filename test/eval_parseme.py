import json
import requests
import sys
import os

LEXICON_PATH = "../dic/lexicon.jsonl"
CUPT_PATH = "../tmp/parseme_en.cupt"
PARSEME_URL = "https://gitlab.com/parseme/sharedtask-data/-/raw/master/1.1/EN/test.cupt"
URL = "http://localhost:3001/parse"

def ensure_dataset_exists():
    if not os.path.exists(CUPT_PATH):
        print(f"Dataset not found at {CUPT_PATH}. Downloading from GitLab...")
        os.makedirs(os.path.dirname(CUPT_PATH), exist_ok=True)
        response = requests.get(PARSEME_URL, stream=True)
        response.raise_for_status()
        with open(CUPT_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

def load_lexicon_phrases():
    phrases = set()
    with open(LEXICON_PATH, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            phrases.add(entry["phrase"].lower())
    return phrases

def parse_cupt(filepath):
    sentences = []
    current_text = ""
    current_tokens = [] # (id, word, lemma)
    current_mwes = {}   # mwe_id -> { "token_ids": set(), "lemmas": [] }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    mwes_list = []
                    for m_id, m_data in current_mwes.items():
                        lemma_phrase = " ".join(m_data["lemmas"]).lower()
                        mwes_list.append({
                            "token_ids": m_data["token_ids"],
                            "lemma_phrase": lemma_phrase
                        })
                    sentences.append({
                        "text": current_text,
                        "mwes": mwes_list
                    })
                current_tokens = []
                current_text = ""
                current_mwes = {}
                continue

            if line.startswith("# text = "):
                current_text = line[len("# text = "):]
            elif line.startswith("#"):
                continue
            else:
                parts = line.split('\t')
                if '-' in parts[0] or '.' in parts[0]:
                    # Skip multi-word tokens in CoNLL-U format (e.g. 1-2)
                    continue

                if len(parts) >= 11:
                    t_id = int(parts[0])
                    word = parts[1]
                    lemma = parts[2]
                    mwe_col = parts[10]
                    current_tokens.append((t_id, word, lemma))

                    if mwe_col != "*":
                        mwe_parts = mwe_col.split(';')
                        for mwe_part in mwe_parts:
                            if mwe_part != '*':
                                mwe_id = mwe_part.split(':')[0]
                                if mwe_id not in current_mwes:
                                    current_mwes[mwe_id] = {"token_ids": set(), "lemmas": []}
                                current_mwes[mwe_id]["token_ids"].add(t_id)
                                current_mwes[mwe_id]["lemmas"].append(lemma)

    return sentences

def evaluate(sentences, lexicon_phrases):
    total_expected = 0
    total_predicted = 0
    true_positives = 0

    report_lines = []

    # We batch requests to speed things up
    batch_size = 50
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        texts = [s["text"] for s in batch]

        try:
            response = requests.post(URL, json={"sentences": texts}, stream=True)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                continue

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')

                if line.startswith('data: '):
                    data = json.loads(line[6:])

                    if 'index' in data and 'result' in data:
                        idx = data['index']
                        res = data['result']
                        sent = batch[idx]
                        sent_text = sent["text"]

                        expected_mwes = []
                        for m in sent["mwes"]:
                            # IMPORTANT: We only expect the parser to find it if it's in our lexicon!
                            if m["lemma_phrase"] in lexicon_phrases:
                                expected_mwes.append(m)

                        predicted_mwes = res.get("mwes", [])

                        # Match using frozen sets of token IDs
                        expected_sets = {frozenset(m["token_ids"]): m for m in expected_mwes}
                        predicted_sets = {frozenset(m["token_ids"]): m for m in predicted_mwes}

                        tp_list = []
                        fp_list = []
                        fn_list = []

                        for p_set, p_data in predicted_sets.items():
                            total_predicted += 1
                            if p_set in expected_sets:
                                true_positives += 1
                                tp_list.append((p_data, expected_sets[p_set]))
                            else:
                                fp_list.append(p_data)

                        for e_set, e_data in expected_sets.items():
                            if e_set not in predicted_sets:
                                fn_list.append(e_data)

                        total_expected += len(expected_mwes)

                        # Append to report if there's any activity
                        if tp_list or fp_list or fn_list:
                            report_lines.append(f"Sentence: {sent_text}")
                            for p, e in tp_list:
                                report_lines.append(f"  [+] TRUE POSITIVE: '{p['surface']}' (lemmas: {e['lemma_phrase']})")
                            for p in fp_list:
                                report_lines.append(f"  [-] FALSE POSITIVE: '{p['surface']}' (lexicon phrase: {p.get('phrase', '')})")
                            for e in fn_list:
                                report_lines.append(f"  [!] FALSE NEGATIVE (Missed): '{e['lemma_phrase']}' (tokens: {sorted(list(e['token_ids']))})")
                            report_lines.append("")

            sys.stdout.write(f"\rProcessed {min(i+batch_size, len(sentences))}/{len(sentences)}")
            sys.stdout.flush()

        except Exception as e:
            print(f"Request failed: {e}")
            break

    precision = true_positives / total_predicted if total_predicted > 0 else 0
    recall = true_positives / total_expected if total_expected > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (
        "--- Evaluation Results ---\n"
        f"Total Expected (In Lexicon): {total_expected}\n"
        f"Total Predicted: {total_predicted}\n"
        f"True Positives (Exact Match): {true_positives}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        "--------------------------\n\n"
    )
    print("\n" + metrics_text)

    report_path = "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(metrics_text)
        f.write("\n".join(report_lines))
    print(f"Detailed report written to: {report_path}")

if __name__ == "__main__":
    print("Loading Lexicon...")
    lexicon_phrases = load_lexicon_phrases()
    print(f"Loaded {len(lexicon_phrases)} phrases from lexicon.")

    ensure_dataset_exists()

    print("Loading PARSEME CUPT...")
    sentences = parse_cupt(CUPT_PATH)
    print(f"Loaded {len(sentences)} sentences.")

    print("Running Evaluation (this might take a minute)...")
    evaluate(sentences, lexicon_phrases)
