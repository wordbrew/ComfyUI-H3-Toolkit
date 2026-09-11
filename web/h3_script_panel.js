// H3 Script — an editor for a take, so nothing has to be remembered.
//
// WHAT THIS REPLACES
//   The script text is a small language: `@ada = character Ada`, `shot | ...`,
//   `say @ada whispers ...`, `---`. It compiles correctly and it is miserable to
//   hold in your head. Every construct here is a button or a dropdown instead:
//   you pick a person from a list, you pick a tone from a list, you press
//   "Someone speaks". There is no syntax to get wrong.
//
// WHY THE TEXT STILL EXISTS
//   The panel writes the script back into the node's widget. That widget is what
//   lands in the workflow JSON, which lands in the mp4's metadata — which is how
//   any render gets diagnosed weeks later. If the panel kept the take somewhere
//   of its own, the video would stop carrying its own story. So: the panel is
//   what you use, the text is only how it is stored.
//
// THE SERVER OWNS THE FORMAT
//   Parsing and writing the text happen in h3script.py over /h3_toolkit/script.
//   This file never composes the language itself — one implementation, and a
//   panel that cannot drift from the compiler.
import { app } from "../../scripts/app.js";

const API = "/h3_toolkit";
const TONES = ["says", "whispers", "moans", "asks", "shouts", "sighs"];
const WIDGET = "script";

async function post(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

// ---------------------------------------------------------------- elements --
function el(tag, props = {}, kids = []) {
  const n = Object.assign(document.createElement(tag), props);
  if (props.style) n.style.cssText = props.style;
  for (const k of kids) n.append(k);
  return n;
}
const BTN = "background:#3a3a3a;color:#ddd;border:1px solid #555;border-radius:4px;" +
            "padding:4px 10px;cursor:pointer;font-size:12px;";
const FIELD = "background:#262626;color:#eee;border:1px solid #444;border-radius:4px;" +
              "padding:5px 7px;font-size:13px;font-family:inherit;";
const CARD = "background:#2b2b2b;border:1px solid #444;border-radius:6px;" +
             "padding:10px;margin-bottom:10px;";

function input(value, placeholder, style = "") {
  return el("input", { value: value ?? "", placeholder, style: FIELD + style });
}
function select(options, value, style = "") {
  const s = el("select", { style: FIELD + style });
  for (const o of options) {
    const label = typeof o === "string" ? o : o.label;
    const val = typeof o === "string" ? o : o.value;
    s.append(el("option", { value: val, textContent: label }));
  }
  s.value = value ?? (typeof options[0] === "string" ? options[0] : options[0]?.value);
  return s;
}
function row(kids, style = "") {
  return el("div", { style: "display:flex;gap:6px;align-items:center;margin-bottom:6px;" + style }, kids);
}
function heading(text) {
  return el("div", { textContent: text, style: "color:#888;font-size:11px;margin:10px 0 4px;letter-spacing:.5px;" });
}

// ------------------------------------------------------------------ panel ---
function openPanel(node) {
  const widget = node.widgets?.find((w) => w.name === WIDGET);
  if (!widget) return;

  let doc = null;
  let characters = [];

  const back = el("div", { style:
    "position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:10000;" +
    "display:flex;align-items:center;justify-content:center;" });
  const panel = el("div", { style:
    "background:#1e1e1e;color:#ddd;border:1px solid #555;border-radius:8px;" +
    "width:min(860px,94vw);max-height:88vh;display:flex;flex-direction:column;" +
    "font-family:system-ui,sans-serif;box-shadow:0 10px 40px rgba(0,0,0,.5);" });
  const head = el("div", { style:
    "padding:12px 16px;border-bottom:1px solid #444;display:flex;" +
    "align-items:center;justify-content:space-between;" }, [
    el("div", { textContent: "The take", style: "font-size:15px;font-weight:600;" }),
    el("div", { textContent: "Nothing here needs remembering — add people, then shots.",
                style: "color:#888;font-size:11px;" }),
  ]);
  const body = el("div", { style: "padding:14px 16px;overflow:auto;flex:1;" });
  const status = el("div", { style: "color:#8a8;font-size:11px;white-space:pre-wrap;flex:1;" });
  const foot = el("div", { style:
    "padding:10px 16px;border-top:1px solid #444;display:flex;gap:8px;align-items:center;" });

  const castBox = el("div");
  const shotBox = el("div");

  // ---- cast -------------------------------------------------------------- //
  function castRow(entry) {
    const wrap = el("div", { style: CARD });
    const name = input(entry.name, "name, e.g. Ada", "width:130px;");
    const kind = select([
      { value: "character", label: "Saved character" },
      { value: "subject", label: "Described person" },
      { value: "setting", label: "Place" },
    ], entry.kind, "width:150px;");
    const picked = select(characters.length ? characters : ["(none saved)"],
                          entry.character, "flex:1;");
    const described = input(entry.description, "what they look like, or the place",
                            "flex:1;");
    const del = el("button", { textContent: "Remove", style: BTN });

    function sync() {
      const isChar = kind.value === "character";
      picked.style.display = isChar ? "" : "none";
      described.style.display = isChar ? "none" : "";
    }
    kind.onchange = sync; sync();
    del.onclick = () => { wrap.remove(); refresh(); };
    for (const n of [name, kind, picked, described]) n.onchange = refresh;

    wrap.append(row([name, kind, picked, described, del]));
    wrap._read = () => {
      const nm = (name.value || "").trim().replace(/[^A-Za-z0-9_]/g, "_");
      if (!nm) return null;
      if (kind.value === "character") {
        return { name: nm, kind: "character", character: picked.value };
      }
      return { name: nm, kind: kind.value, description: described.value };
    };
    return wrap;
  }

  // ---- shots ------------------------------------------------------------- //
  function beatRow(kind, data) {
    const wrap = el("div", { style: "margin-bottom:6px;" });
    const del = el("button", { textContent: "✕", style: BTN + "padding:4px 8px;" });
    del.onclick = () => { wrap.remove(); refresh(); };
    let read;

    if (kind === "say") {
      const names = currentNames();
      const who = select(names.length ? names : ["(add a person)"], data?.who, "width:120px;");
      const tone = select(TONES, data?.verb || "says", "width:110px;");
      const words = input(data?.line, "what they say", "flex:1;");
      for (const n of [who, tone, words]) n.onchange = refresh;
      wrap.append(row([el("span", { textContent: "speaks", style: "color:#888;width:56px;font-size:11px;" }),
                       who, tone, words, del]));
      read = () => ({ type: "say", who: who.value, verb: tone.value, line: words.value });
    } else if (kind === "do") {
      const what = input(data, "what happens — an action, a camera move", "flex:1;");
      what.onchange = refresh;
      wrap.append(row([el("span", { textContent: "action", style: "color:#888;width:56px;font-size:11px;" }),
                       what, del]));
      read = () => ({ type: "do", text: what.value });
    } else if (kind === "note") {
      const what = input(data, "a detail to hold — position, framing", "flex:1;");
      what.onchange = refresh;
      wrap.append(row([el("span", { textContent: "note", style: "color:#888;width:56px;font-size:11px;" }),
                       what, del]));
      read = () => ({ type: "note", text: what.value });
    } else {   // a split, kept because a loaded take may already have one
      wrap.append(row([el("div", { textContent: "— split —",
        style: "flex:1;color:#a88;font-size:11px;border-top:1px dashed #665;padding-top:6px;" }), del]));
      read = () => ({ type: "split" });
    }
    wrap._read = read;
    return wrap;
  }

  function shotCard(shot) {
    const wrap = el("div", { style: CARD });
    const what = input(shot?.description, "what this shot is — framing, who is in it", "flex:1;");
    what.onchange = refresh;
    const del = el("button", { textContent: "Remove shot", style: BTN });
    const beats = el("div", { style: "margin:8px 0 6px;" });

    // a loaded take carries notes, actions, lines and splits; show them in order
    for (const n of shot?.notes || []) beats.append(beatRow("note", n));
    (shot?.chunks || []).forEach((c, i) => {
      if (i) beats.append(beatRow("split"));
      for (const a of c.actions || []) beats.append(beatRow("do", a));
      for (const l of c.lines || []) beats.append(beatRow("say", l));
    });

    const add = (label, kind) => {
      const b = el("button", { textContent: label, style: BTN });
      b.onclick = () => { beats.append(beatRow(kind)); refresh(); };
      return b;
    };
    del.onclick = () => { wrap.remove(); refresh(); };

    wrap.append(row([el("span", { textContent: "shot", style: "color:#888;width:56px;font-size:11px;" }),
                     what, del]));
    wrap.append(beats);
    wrap.append(row([add("Someone speaks", "say"), add("Something happens", "do"),
                     add("A detail to hold", "note"), add("Split here", "split")],
                    "margin-bottom:0;"));

    wrap._read = () => {
      const out = { description: what.value, notes: [],
                    chunks: [{ lines: [], actions: [] }], loras: shot?.loras || [] };
      for (const b of beats.children) {
        const v = b._read?.();
        if (!v) continue;
        if (v.type === "note") { if (v.text) out.notes.push(v.text); }
        else if (v.type === "split") out.chunks.push({ lines: [], actions: [] });
        else {
          const cur = out.chunks[out.chunks.length - 1];
          if (v.type === "do" && v.text) cur.actions.push(v.text);
          if (v.type === "say" && v.line) {
            cur.lines.push({ who: v.who, verb: v.verb, line: v.line });
          }
        }
      }
      return out;
    };
    return wrap;
  }

  function currentNames() {
    return [...castBox.children].map((c) => c._read?.()?.name).filter(Boolean);
  }

  function build() {
    const cast = [...castBox.children].map((c) => c._read?.()).filter(Boolean);
    const shots = [...shotBox.children].map((c) => c._read?.()).filter(Boolean);
    return { ...(doc || {}), version: 1, cast, shots };
  }

  // live numbering + lint, without queueing anything
  let pending = null;
  function refresh() {
    clearTimeout(pending);
    pending = setTimeout(async () => {
      const d = build();
      if (!d.cast.length) { status.textContent = "Add a person to begin."; return; }
      const r = await post("/script/compile", { document: d });
      if (!r.ok) { status.textContent = r.error; status.style.color = "#d88"; return; }
      status.style.color = "#8a8";
      const who = d.cast.map((c) => `${c.name} = Subject ${r.index?.[c.name]?.subject ?? "?"}`);
      status.textContent = who.join("   ") +
        (r.lint?.length ? "\n" + r.lint.join("\n") : "");
    }, 150);
  }

  // ---- wiring ------------------------------------------------------------ //
  const addPerson = el("button", { textContent: "Add a person", style: BTN });
  addPerson.onclick = () => { castBox.append(castRow({ kind: "character" })); refresh(); };
  const addPlace = el("button", { textContent: "Add a place", style: BTN });
  addPlace.onclick = () => { castBox.append(castRow({ kind: "setting" })); refresh(); };
  const addShot = el("button", { textContent: "Add a shot", style: BTN });
  addShot.onclick = () => { shotBox.append(shotCard()); refresh(); };

  body.append(heading("PEOPLE AND PLACES"), castBox, row([addPerson, addPlace]),
              heading("SHOTS"), shotBox, row([addShot]));

  const save = el("button", { textContent: "Save", style: BTN + "background:#2d5a2d;" });
  const cancel = el("button", { textContent: "Cancel", style: BTN });
  cancel.onclick = () => back.remove();
  save.onclick = async () => {
    const r = await post("/script/serialize", { document: build() });
    if (!r.ok) { status.textContent = r.error; status.style.color = "#d88"; return; }
    widget.value = r.text;
    if (widget.inputEl) widget.inputEl.value = r.text;   // multiline DOM widget
    widget.callback?.(r.text);
    app.graph.setDirtyCanvas(true, true);
    back.remove();
  };
  foot.append(status, cancel, save);
  panel.append(head, body, foot);
  back.append(panel);
  back.onclick = (e) => { if (e.target === back) back.remove(); };
  document.body.append(back);

  // ---- load the current take --------------------------------------------- //
  (async () => {
    try {
      const c = await (await fetch(API + "/characters")).json();
      characters = c.characters || [];
    } catch { characters = []; }
    const r = await post("/script/parse", { text: widget.value || "" });
    if (r.ok) doc = r.document;
    else status.textContent = r.error;   // show it, keep the panel usable
    for (const entry of doc?.cast || []) castBox.append(castRow(entry));
    for (const shot of doc?.shots || []) shotBox.append(shotCard(shot));
    refresh();
  })();
}

app.registerExtension({
  name: "h3.script.panel",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3Script") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      this.addWidget("button", "Edit the take", null, () => openPanel(this));
    };
  },
});
