// Tiny Deno client/smoke-test for the /parse SSE endpoint.
//
//   deno run --allow-net data/parse.ts "He finally spilled the beans." "I made up my mind."
//
// No deps. Streams the SSE response, prints progress, then the full parse
// (tokens / phrasal_verbs / idioms) per sentence. Exits non-zero on error or
// if any sentence yields no tokens — usable as a CI smoke check.

const URL = Deno.env.get("PARSE_URL") ?? "http://localhost:3000/parse";

const sentences = Deno.args.length > 0 ? Deno.args : [
  "He finally spilled the beans about the party.",
  "Mike got out of the water and made up his mind.",
];

type Token = { id: number; word: string; head: number; rel: string; upos: string };
type Pv = {
  verb: string; particle: string; verb_id: number; particle_id: number;
  // 3-component verbs ("come up with"): present only when the extending
  // preposition was found in the tree (server omits it otherwise).
  prep?: string; prep_id?: number;
};
type Idiom = {
  surface: string; words: string[]; prob: number;
  // Readable contiguous span incl. fillers ("costs an arm and a leg");
  // optional for back-compat with servers that predate the field.
  span_text?: string;
  idiomatic: boolean; discontinuous: boolean; tree_connected: boolean;
};
type Sentence = { tokens: Token[]; phrasal_verbs: Pv[]; idioms: Idiom[] };

let res: Response;
try {
  res = await fetch(URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ sentences }),
  });
} catch (e) {
  console.error(
    `Cannot reach ${URL}: ${e instanceof Error ? e.message : e}\n` +
      `Is the service running?  ./target/release/parser-service`,
  );
  Deno.exit(1);
}
if (!res.ok || !res.body) {
  console.error(`HTTP ${res.status}`);
  Deno.exit(1);
}

// Minimal SSE reader: split the byte stream into "\n\n"-delimited events,
// pull `event:` / `data:` lines out of each.
let done: { results: Sentence[] } | null = null;
let buf = "";
const decoder = new TextDecoder();
for await (const chunk of res.body) {
  buf += decoder.decode(chunk, { stream: true });
  let sep: number;
  while ((sep = buf.indexOf("\n\n")) !== -1) {
    const block = buf.slice(0, sep);
    buf = buf.slice(sep + 2);
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (event === "progress") {
      const p = JSON.parse(data);
      console.log(`  progress: ${p.done}/${p.total} (${p.percent}%)`);
    } else if (event === "error") {
      console.error(`Error: ${data}`);
      Deno.exit(1);
    } else if (event === "done") {
      done = JSON.parse(data);
    }
  }
}

if (!done) {
  console.error("stream closed without a `done` event");
  Deno.exit(1);
}

let ok = true;
done.results.forEach((s, i) => {
  console.log(`\nSENT ${i}: ${sentences[i]}`);
  if (s.tokens.length === 0) ok = false;
  for (const t of s.tokens) {
    console.log(
      `  ${String(t.id).padStart(2)} ${t.word.padEnd(12)} ` +
        `head=${t.head} ${t.rel.padEnd(12)} ${t.upos}`,
    );
  }
  for (const pv of s.phrasal_verbs) {
    const prep = pv.prep ? ` + ${pv.prep}` : "";
    const ids = pv.prep_id !== undefined
      ? `(${pv.verb_id},${pv.particle_id},${pv.prep_id})`
      : `(${pv.verb_id},${pv.particle_id})`;
    console.log(`  phrasal: ${pv.verb} + ${pv.particle}${prep} ${ids}`);
  }
  for (const id of s.idioms) {
    console.log(
      `  idiom: "${id.span_text ?? id.surface}" ` +
        `(lex: ${id.surface}) ` +
        `p=${id.prob.toFixed(3)} ${id.idiomatic ? "IDIOMATIC" : "literal"}` +
        `${id.discontinuous ? (id.tree_connected ? " disc/connected" : " disc/REJECT") : ""}`,
    );
  }
});

console.log(
  `\nDone: ${done.results.length} sentences, ` +
    `${done.results.reduce((n, s) => n + s.tokens.length, 0)} tokens`,
);
Deno.exit(ok ? 0 : 1);
