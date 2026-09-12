// Does the panel actually RUN? — a headless open, with the DOM stubbed.
//
// WHY THIS EXISTS
//   h3_script_panel.js was built and then edited perhaps fifteen times by blind
//   string replacement. `node --check` proves it parses; it cannot see a
//   replacement that failed to match and left a call to a function since
//   deleted. One did exactly that -- the dialogue lane still called `laneLabel`
//   and `LANE` after both were removed -- and the panel would have thrown the
//   moment it was opened, with nothing before that point catching it.
//
//   So: stub enough DOM to execute, open the panel, click things, and fail on
//   any throw. This is not a test of how it LOOKS. It is a test that every path
//   through it still resolves.
//
//     node test_panel_smoke.js
const fs = require("fs");
const path = require("path");

const failures = [];

// drawBoard runs inside an async refresh, so anything it throws becomes an
// unhandled rejection and vanishes — which is exactly how a half-drawn board
// looks like a working one. Make it loud.
process.on("unhandledRejection", (err) => {
  console.log(`  FAIL swallowed inside an async path: ${err && err.stack
    ? err.stack.split("\n").slice(0, 3).join("\n       ") : err}`);
  failures.push(`unhandled rejection: ${err && err.message}`);
});
const check = (label, fn) => {
  try { fn(); console.log(`  ok   ${label}`); }
  catch (e) { failures.push(`${label}: ${e.message}`);
              console.log(`  FAIL ${label}: ${e.message}`); }
};

// ---- the smallest DOM that lets the panel run ---------------------------- //
function makeEl(tag) {
  const node = {
    tagName: tag, children: [], dataset: {}, value: "",
    textContent: "", innerHTML: "", className: "", title: "", type: "",
    placeholder: "", hidden: false, parentNode: null,
    append(...kids) {
      for (const k of kids) {
        if (k == null) continue;
        k.parentNode = this;
        this.children.push(k);
      }
    },
    appendChild(k) { this.append(k); return k; },
    insertBefore(k, ref) {
      const i = ref ? this.children.indexOf(ref) : this.children.length;
      this.children.splice(i < 0 ? this.children.length : i, 0, k);
      k.parentNode = this; return k;
    },
    removeChild(k) {
      const i = this.children.indexOf(k);
      if (i >= 0) this.children.splice(i, 1);
      return k;
    },
    remove() { this.parentNode?.removeChild(this); },
    after(k) { this.parentNode?.append(k); },
    setAttribute() {}, getAttribute() { return null; },
    addEventListener() {}, removeEventListener() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 60 }; },
    scrollIntoView() {},
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
    querySelectorAll(sel) {
      // enough selector support for what the panel actually asks for
      const want = sel.replace(":scope", "").trim();
      const out = [];
      const walk = (n, depth) => {
        for (const c of n.children) {
          if (want.startsWith(".") && c.className === want.slice(1)) out.push(c);
          else if (want.startsWith("> div") && depth === 0 && c.tagName === "div") out.push(c);
          else if (want === "span" && c.tagName === "span") out.push(c);
          walk(c, depth + 1);
        }
      };
      walk(this, 0);
      return out;
    },
  };
  // A BROWSER ACCEPTS `el.style = "a:b"` and routes it to cssText. A plain
  // property does not, so the panel's own el() helper -- Object.assign first,
  // then style.cssText -- silently lost every style in this harness and made a
  // real control look undrawn.
  const decl = { cssText: "" };
  Object.defineProperty(node, "style", {
    get() { return decl; },
    set(v) { decl.cssText = typeof v === "string" ? v : String(v); },
  });
  Object.defineProperty(node, "firstChild",
    { get() { return this.children[0] || null; } });
  return node;
}

const body = makeEl("body");
global.document = {
  body,
  createElement: makeEl,
  getElementById: () => null,
  addEventListener() {}, removeEventListener() {},
  get activeElement() { return null; },
};
global.window = global;

// canned server answers, shaped like the real routes
const TIMING = {
  ok: true, total_frames: 384, cuts: [141],
  chunks: [
    { index: 0, shot: 0, start: 0, frames: 141, pin: 0, pin_s: 0,
      run_s: 5.9, speech_from: 0.6, speech_to: 5.3, starts_at: 0.6 },
    { index: 1, shot: 1, start: 141, frames: 102, pin: 39, pin_s: 1.6,
      run_s: 5.9, speech_from: 1.7, speech_to: 5.3, starts_at: 5.9 },
  ],
  shots: [
    { index: 0, start: 0, frames: 141, requested: 141, seconds: 5.9 },
    { index: 1, start: 141, frames: 243, requested: 250, seconds: 10.1 },
  ],
  lines: [
    { shot: 0, who: "ada", line: "I told you I would come back.", chunk: 0,
      start: 0.6, local: 0.6, seconds: 1.6, placed: false, problem: null },
    { shot: 1, who: "ada", line: "Do you want to play?", chunk: 1,
      start: 5.9, local: 1.7, seconds: 1.1, placed: true,
      problem: "starts inside the 1.62s carried handle — never said" },
  ],
  problems: ["the carried handles cost 1.6s of the take"],
  sayable_seconds: 8.3, pinned_seconds: 1.6,
};
const DOCUMENT = {
  ok: true,
  document: {
    version: 1, task: "reference generation", soundscape: "", music: "N/A",
    cast: [{ name: "ada", kind: "character", character: "skye" },
           { name: "room", kind: "setting", description: "a bedroom" }],
    shots: [
      { description: "close on her at the window", frames: 141, notes: ["nearer camera"],
        loras: [{ name: "h3/A.safetensors", strength: "0.8" }],
        chunks: [{ lines: [{ who: "ada", verb: "whispers",
                             line: "I told you I would come back." }],
                   actions: ["rain on the glass"] }] },
      { description: "wider, she crosses the room", frames: 243, notes: [],
        loras: [],
        chunks: [{ lines: [{ who: "ada", verb: "asks", at: 1.7,
                             line: "Do you want to play?" }], actions: [] }] },
    ],
  },
};
global.fetch = async (url) => ({
  json: async () => {
    if (url.includes("/characters")) return { ok: true, characters: ["skye", "Ada"] };
    if (url.includes("/script/timing")) return TIMING;
    if (url.includes("/script/parse")) return DOCUMENT;
    if (url.includes("/script/serialize")) return { ok: true, text: "@ada = character skye\n" };
    if (url.includes("/script/compile"))
      return { ok: true, index: { ada: { subject: 1 }, room: { subject: 2 } },
               counts: { subjects: 2 }, lint: [], fields: { head: "a head" } };
    return { ok: false, error: "unrouted: " + url };
  },
});

// ---- the extension host -------------------------------------------------- //
const registered = [];
const app = {
  registerExtension: (ext) => registered.push(ext),
  graph: { _nodes: [], setDirtyCanvas() {} },
};

const src = fs.readFileSync(path.join(__dirname, "web", "h3_script_panel.js"), "utf8")
  .replace(/^import[^\n]*\n/m, "");           // the one ESM line
const load = new Function("app", "document", "window", "fetch", src + "\n;return true;");

console.log("the module evaluates and registers its extension");
check("no throw while loading", () => {
  if (!load(app, global.document, global, global.fetch)) throw new Error("no return");
});
check("one extension registered", () => {
  if (registered.length !== 1) throw new Error(`got ${registered.length}`);
  if (registered[0].name !== "h3.script.panel")
    throw new Error(`named ${registered[0].name}`);
});

// ---- open the panel ------------------------------------------------------ //
const widgets = [{ name: "script", value: "@ada = character skye\n", callback() {} },
                 { name: "chunk_frames", value: 141, callback() {} },
                 { name: "context", value: 39, callback() {} }];
let openPanelFn = null;
const fakeNode = {
  widgets,
  addWidget(type, name, value, cb) { openPanelFn = cb; return { name }; },
};

console.log("opening it runs every draw path");
check("the node gains its button", () => {
  registered[0].beforeRegisterNodeDef.call(null,
    { prototype: { onNodeCreated: null } }, { name: "H3Script" });
});

// call the registered prototype hook the way ComfyUI would
const proto = { onNodeCreated: null };
registered[0].beforeRegisterNodeDef.call(null, { prototype: proto }, { name: "H3Script" });
check("onNodeCreated installs a button", () => {
  proto.onNodeCreated.call(fakeNode);
  if (typeof openPanelFn !== "function") throw new Error("no callback registered");
});

check("openPanel runs without throwing", () => { openPanelFn(); });

// the load is async AND refresh() debounces by 150ms, so the board does not
// exist for a moment after opening. Waiting 50ms once produced a confident
// "no Chunks lane" that was entirely my harness being impatient.
setTimeout(() => {
  check("the panel attached itself to the page", () => {
    if (!body.children.length) throw new Error("nothing appended to body");
  });
  check("the board drew lanes", () => {
    const text = JSON.stringify(collect(body));
    for (const name of ["Shots", "Chunks", "Dialogue", "Action"]) {
      if (!text.includes(name)) throw new Error(`no ${name} lane`);
    }
  });
  check("a problem line is carried through to the view", () => {
    if (!JSON.stringify(collect(body)).includes("carried handle"))
      throw new Error("the warning never reached the DOM");
  });
  check("a snapped shot shows what was asked for", () => {
    if (!JSON.stringify(collect(body)).includes("250"))
      throw new Error("the requested length is not shown");
  });

  if (process.env.DUMP) {
    const flat = (n, d = 0) => {
      const bits = [n.textContent, n.value].filter(Boolean).join(" | ");
      if (bits) console.log("  ".repeat(d) + bits.slice(0, 70));
      for (const c of n.children) flat(c, d + 1);
    };
    flat(body);
  }
  // ---- the buttons ------------------------------------------------------ //
  // Every one of these is a path my string-replacement edits could have left
  // pointing at something deleted. Clicking them is the only way to know.
  const find = (label) => {
    let hit = null;
    const walk = (n) => {
      if (!hit && n.textContent === label && typeof n.onclick === "function") hit = n;
      for (const c of n.children) walk(c);
    };
    walk(body);
    return hit;
  };
  const click = (label) => {
    const b = find(label);
    if (!b) throw new Error(`no control labelled "${label}"`);
    b.onclick({ target: b, preventDefault() {}, stopPropagation() {} });
  };

  console.log("every control resolves when clicked");
  for (const label of ["Add a reference", "Add a shot"]) {
    check(label, () => click(label));
  }
  check("switching a reference to 'Described person' swaps in a text field", () => {
    // the handler for this was overwritten by a loop that set refresh on every
    // control, so the picker never went away and there was nowhere to type
    let kind = null;
    const walk = (n) => {
      if (!kind && n.tagName === "select" &&
          n.children.some((o) => o.textContent === "Described person")) kind = n;
      for (const c of n.children) walk(c);
    };
    walk(body);
    if (!kind) throw new Error("no reference-kind control drawn");
    if (typeof kind.onchange !== "function") throw new Error("kind has no handler");
    kind.value = "subject";
    kind.onchange();
    const rowEl = kind.parentNode;
    const textField = rowEl.children.find(
      (c) => c.tagName === "input" && (c.style.cssText || "").indexOf("none") < 0 &&
             c.placeholder && c.placeholder.indexOf("describe") >= 0);
    if (!textField) throw new Error("no visible description field after switching");
  });
  for (const label of ["Someone speaks", "Something happens",
                       "A detail to hold", "A LoRA for this shot"]) {
    check(label, () => click(label));
  }
  check("Show the prompt", () => click("Show the prompt"));
  check("Save writes the script back to the widget", () => {
    click("Save");
    const w = widgets.find((x) => x.name === "script");
    if (!w.value.includes("@ada")) throw new Error("the widget was not written");
  });

  console.log("dragging and undo resolve");
  check("resizing a shot", () => {
    const grips = [];
    // the panel sets style.cssText, not individual properties, so look there
    const walk = (n) => {
      if ((n.style.cssText || "").includes("ew-resize")) grips.push(n);
      for (const c of n.children) walk(c);
    };
    walk(body);
    if (!grips.length) throw new Error("no resize grips drawn");
    grips[0].onmousedown({ clientX: 10, target: grips[0],
                           preventDefault() {}, stopPropagation() {} });
  });
  check("placing a line", () => {
    let beat = null;
    const walk = (n) => { if (!beat && typeof n.ondblclick === "function") beat = n;
                          for (const c of n.children) walk(c); };
    walk(body);
    if (!beat) throw new Error("no draggable line drawn");
    beat.onmousedown({ clientX: 40, target: beat,
                       preventDefault() {}, stopPropagation() {} });
    beat.ondblclick();
  });

  console.log();
  console.log(failures.length
    ? `${failures.length} failure(s)`
    : "panel smoke: it opens, draws and resolves every call");
  process.exit(failures.length ? 1 : 0);
}, 600);

function collect(node) {
  const out = [node.textContent || "", node.value || "", node.title || ""];
  for (const c of node.children) out.push(collect(c));
  return out;
}
