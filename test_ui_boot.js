// Boot ui/index.html under a minimal DOM, with EVERY fetch failing, and assert
// the controls are still wired.
//
// THE BUG THIS PINS: init() awaited its loading before attaching handlers, so a
// throw anywhere in that phase aborted it and the Render button was never wired.
// CJ pressed Render and "resulted in nothing" -- no job, no error, no clue,
// because there was no handler to run. A page that cannot reach ComfyUI should
// degrade to empty dropdowns, which can be seen and reported; it must never
// degrade to dead buttons.
//
// The script is read out of ui/index.html, so this cannot pass against a stale
// copy of init().
// A throw during init leaves the Render handler unattached, and the button then
// does nothing at all -- which is exactly "hitting render resulted in nothing".
const fs = require("fs");
const html = fs.readFileSync(__dirname + "/ui/index.html", "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const ids = [...html.matchAll(/id="([a-zA-Z0-9_]+)"/g)].map(m => m[1]);

const mk = (id) => {
  const el = {
    id, value: "", textContent: "", innerHTML: "", hidden: false, disabled: false,
    options: [], classList: { add(){}, remove(){}, contains(){return false} },
    dataset: {}, style: {}, title: "", open: false, selectedOptions: [],
    appendChild(c){ this.options.push(c); return c; },
    querySelector(){ return mk("q"); }, querySelectorAll(){ return []; },
    addEventListener(t){ WIRED.add(id + ":" + t); }, removeEventListener(){},
    showModal(){}, close(){},
    getBoundingClientRect(){ return {left:0,top:0,right:0,bottom:0,width:800,height:20}; },
    focus(){}, click(){}, remove(){},
  };
  return el;
};
global.WIRED = new Set();
const store = {};
for (const id of ids) store[id] = mk(id);
global.document = {
  getElementById: (id) => store[id] || null,
  createElement: () => mk("new"),
  body: mk("body"), documentElement: { dataset: {} },
  querySelectorAll: () => [], addEventListener(){},
};
global.window = { addEventListener(){}, open(){ return { document: { write(){} } }; } };
global.location = { protocol: "http:", host: "127.0.0.1:8188" };
global.localStorage = { getItem(){ return null; }, setItem(){} };
// TWO SCENARIOS, and the second is the one that was missing. Scenario "fail"
// proves the controls survive an unreachable ComfyUI. Scenario "ok" proves init
// RUNS TO COMPLETION when everything is reachable -- which is where
// "paintStatic is not defined" lived: a reference only reached on the happy
// path, so a test that only broke the network never saw it.
// Run both by default; SCENARIO=ok|fail picks one. A bare invocation has to
// cover the reachable path, because that is where the last bug lived.
const SCENARIO = process.env.SCENARIO || "fail";
if (!process.env.SCENARIO) {
  const { execFileSync } = require("child_process");
  let rc = 0;
  for (const sc of ["ok", "fail"]) {
    console.log(`--- scenario ${sc}`);
    try {
      execFileSync(process.execPath, [__filename],
                   { env: { ...process.env, SCENARIO: sc }, stdio: "inherit" });
    } catch { rc = 1; }
  }
  process.exit(rc);
}
const okBody = (url) => {
  if (url.includes("/continuity")) {
    return { ok: true, default: "carry", kinds: [
      { id: "carry", label: "Continue the shot", what: "goes on", wired: true },
      { id: "cut", label: "Hard cut", what: "nothing carried", wired: true }] };
  }
  if (url.includes("/object_info")) {
    return { LoadImage: { input: { required: { image: [["a.png", "b.png"]] } } },
             UNETLoader: { input: { required: { unet_name: [["H3/x.safetensors"]] } } },
             CLIPLoader: { input: { required: { clip_name: [["q.safetensors"]] } } },
             VAELoader: { input: { required: { vae_name: [["v.safetensors"]] } } },
             LoraLoaderModelOnly: { input: { required: { lora_name: [["h3/l.safetensors"]] } } },
             LoadAudio: { input: { required: { audio: [["a.wav"]] } } } };
  }
  if (url.includes("/subjects")) return { ok: true, subjects: [] };
  if (url.includes("/characters")) return { ok: true, characters: [] };
  if (url.includes("/take/plan") || url.includes("/take/scenes")) {
    return { ok: true, total_frames: 192, total_seconds: 8, chunk_count: 1,
             chunk_frames: 192, context: 39, beats_required: 1, lint: [],
             scenes: [], chunks: [{ index: 0, number: 1, frames: 192,
                                    start_seconds: 0, end_seconds: 8,
                                    beat: null, continuity: null }] };
  }
  return { ok: true };
};
global.fetch = async (url) => {
  if (SCENARIO === "fail") throw new Error("simulated: ComfyUI unreachable");
  return { ok: true, status: 200, json: async () => okBody(String(url)) };
};
// fail() writes the red box; record that it was reached
global.FAILED_MSG = null;
global.WebSocket = function(){ return { close(){} }; };
global.confirm = () => true;
global.FormData = function(){ this.append = () => {}; };
global.URL = { createObjectURL: () => "blob:", revokeObjectURL(){} };
// THE HARNESS MUST NOT OUTRUN THE PAGE. An earlier version made setTimeout
// synchronous, so the assertions ran before init()'s promise settled and the
// "no error box" check passed because nothing had happened yet -- a vacuous
// pass, which is worse than a failure. Keep the real timer and wait.
const realTimeout = setTimeout;
global.setTimeout = (f, ms) => realTimeout(f, ms || 0);
global.clearTimeout = clearTimeout;

process.on("unhandledRejection", e => {
  console.log("UNHANDLED REJECTION during init:\n  " + (e && e.stack || e));
  process.exit(1);
});
try {
  new Function(src)();
  console.log("script evaluated with no synchronous throw");
} catch (e) {
  console.log("SYNCHRONOUS THROW:\n  " + (e.stack || e));
  process.exit(1);
}
realTimeout(() => {
  // onclick assignments are plain properties, so check those too
  for (const [id, el] of Object.entries(store)) {
    if (typeof el.onclick === "function") WIRED.add(id + ":onclick");
  }
  const need = ["render:click", "i_addlora:onclick",
                "newsubj:onclick", "dry:click",
                // the three timeline + buttons replaced Quick setup's one
                "addchunk:onclick", "addcast2:onclick", "addref:onclick"];
  let bad = 0;
  for (const k of need) {
    const ok = WIRED.has(k);
    console.log(`  ${ok ? "ok  " : "FAIL"} ${k} attached`);
    if (!ok) bad++;
  }
  // On the reachable path init must finish WITHOUT surfacing an error. The red
  // box is written by fail(), which sets #err's textContent and unhides it.
  const err = store["err"];
  const surfaced = err && err.hidden === false && String(err.textContent || "");
  if (SCENARIO === "ok") {
    const clean = !surfaced;
    console.log(`  ${clean ? "ok  " : "FAIL"} init completed with no error box`);
    if (!clean) { bad++; console.log("       " + surfaced.slice(0, 160)); }
  } else {
    // An unreachable server does NOT reach init's catch: loadTransitions(),
    // optionsFor() and loadLibrary() each swallow their own failure by design,
    // so the page degrades to empty dropdowns. The signal is the connection
    // indicator, set by refreshPlan's catch -- that is what must be true, and
    // asserting the red box here was my expectation being wrong, not the code.
    const conn = String(store["conn"].textContent || "");
    const dot = String(store["dot"].className || "");
    const told = conn.includes("no server") || dot.includes("off");
    console.log(`  ${told ? "ok  " : "FAIL"} the connection indicator says ` +
                `disconnected (got ${JSON.stringify(conn)})`);
    if (!told) bad++;
    const lists = store["unet"].options.length;
    console.log(`  ok   and the model list degraded to ${lists} entries`);
  }
  console.log(bad ? `\nFAIL — ${bad} problem(s) in scenario "${SCENARIO}"` :
                    `\nscenario "${SCENARIO}": clean`);
  process.exit(bad ? 1 : 0);
}, 150);
