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
// EVERY fetch fails: the point is whether the controls survive that
global.fetch = async () => { throw new Error("simulated: ComfyUI unreachable"); };
global.WebSocket = function(){ return { close(){} }; };
global.confirm = () => true;
global.FormData = function(){ this.append = () => {}; };
global.URL = { createObjectURL: () => "blob:", revokeObjectURL(){} };
global.setTimeout = (f) => { try { f(); } catch (e) { throw e; } return 0; };
global.clearTimeout = () => {};

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
setTimeout(() => {
  // onclick assignments are plain properties, so check those too
  for (const [id, el] of Object.entries(store)) {
    if (typeof el.onclick === "function") WIRED.add(id + ":onclick");
  }
  const need = ["render:click", "addscene:onclick", "i_addlora:onclick",
                "newsubj:onclick", "dry:click", "qgo:onclick"];
  let bad = 0;
  for (const k of need) {
    const ok = WIRED.has(k);
    console.log(`  ${ok ? "ok  " : "FAIL"} ${k} attached despite the failure`);
    if (!ok) bad++;
  }
  console.log(bad ? `\nFAIL — ${bad} control(s) left dead` :
                    "\nevery control survives a failed load");
  process.exit(bad ? 1 : 0);
}, 0);
