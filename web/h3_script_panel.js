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
// SUGGESTIONS, not a vocabulary. "says quietly, half-turning away" reads nothing
// like "says", and that phrase is real prompt surface — the compiler takes any
// phrase and H3Dialogue always did. A closed dropdown quietly threw that away.
const TONES = ["says", "whispers", "moans", "asks", "shouts", "sighs",
               "says quietly", "almost laughing", "under her breath",
               "flatly", "half-turning away"];
const TONE_LIST = "h3-tones";
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
  const timing = el("div", { style:
    "padding:8px 16px;border-top:1px solid #3a3a3a;background:#242424;" +
    "font-size:11px;color:#aaa;white-space:pre-wrap;max-height:150px;overflow:auto;" });
  const status = el("div", { style: "color:#8a8;font-size:11px;white-space:pre-wrap;flex:1;" });
  const foot = el("div", { style:
    "padding:10px 16px;border-top:1px solid #444;display:flex;gap:8px;align-items:center;" });

  if (!document.getElementById(TONE_LIST)) {
    const dl = el("datalist", { id: TONE_LIST });
    for (const v of TONES) dl.append(el("option", { value: v }));
    document.body.append(dl);
  }

  const castBox = el("div");
  const shotBox = el("div");

  const placed = new Map();

  // THE BOARD. Shots on a real clock, the chunks the planner will make drawn
  // underneath them, and every line at the second it is actually spoken. The
  // point is not tidiness: a carried handle and a chunk boundary are properties
  // of the PLAN, so where a line lands is not guessable while writing it.
  const board = el("div", { style:
    "border:1px solid #3a4148;border-radius:6px;background:#1b1f23;padding:8px 10px 10px;" +
    "margin-bottom:4px;overflow-x:auto;" });
  const boardInner = el("div", { style: "min-width:620px;position:relative;" });
  board.append(boardInner);
  let selShot = 0;

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
      const tone = input(data?.verb || "says", "how they say it", "width:150px;");
      tone.setAttribute("list", TONE_LIST);
      tone.oninput = refresh;
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
    // 17n+5, because a shot IS a cut and the planner will trim anything else --
    // the number you designed to would not be the number rendered.
    const frames = input(shot?.frames ?? 141, "frames", "width:78px;");
    frames.className = "h3-frames";
    frames.onchange = () => {
      const n = Math.max(5, Math.round((Number(frames.value) - 5) / 17) * 17 + 5);
      frames.value = n; wrap._frames = n; refresh();
    };
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
                     what, frames, del]));
    wrap.append(beats);
    wrap.append(row([add("Someone speaks", "say"), add("Something happens", "do"),
                     add("A detail to hold", "note"), add("Split here", "split")],
                    "margin-bottom:0;"));

    wrap._frames = Number(shot?.frames ?? 141);
    wrap._read = () => {
      const out = { description: what.value, notes: [],
                    frames: Number(wrap._frames) || 141,
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
            const l = { who: v.who, verb: v.verb, line: v.line };
            // the form rebuilds the document from the DOM every refresh, so a
            // time placed by dragging has to be reattached or the drag undoes
            // itself on the next keystroke
            if (placed.has(v.line)) l.at = placed.get(v.line);
            cur.lines.push(l);
          }
        }
      }
      return out;
    };
    return wrap;
  }

  // THE PLAN IS IN THE GRAPH, not in this panel. H3ChunkPlan already holds the
  // take's length, chunk size and context; reading them keeps the timing strip
  // describing the render that will actually happen instead of a default.
  function planFromGraph() {
    const n = app.graph?._nodes?.find((x) => x.type === "H3ChunkPlan");
    const get = (name, fallback) => {
      const w = n?.widgets?.find((x) => x.name === name);
      const v = Number(w?.value);
      return Number.isFinite(v) && v > 0 ? v : fallback;
    };
    return {
      total_frames: get("total_frames", 345),
      chunk_frames: get("chunk_frames", 141),
      context: get("context", 39),
    };
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
      drawTiming(d);
    }, 150);
  }

  // A SHOT IS ONE OR MORE CHUNKS, never fewer. Shots become cuts and the planner
  // opens a chunk at each one, so a shot always starts on a boundary — but a shot
  // longer than chunk_frames spans several, which is what a split is for. The
  // editor showed shots and the strip showed chunks with nothing tying them
  // together, so which chunk a line would land in was still a guess.
  function markShots(t) {
    const cards = [...shotBox.children];
    const spans = new Map();
    for (const l of t.lines || []) {
      if (l.chunk === null || l.chunk === undefined) continue;
      const cur = spans.get(l.shot) || [l.chunk, l.chunk];
      spans.set(l.shot, [Math.min(cur[0], l.chunk), Math.max(cur[1], l.chunk)]);
    }
    cards.forEach((card, i) => {
      let tag = card.querySelector(".h3-span");
      if (!tag) {
        tag = el("span", { className: "h3-span",
          style: "font-size:10px;color:#89a;margin-left:6px;white-space:nowrap;" });
        card.querySelector("span")?.after(tag);
      }
      const s = spans.get(i);
      tag.textContent = !s ? "" : s[0] === s[1] ? `chunk ${s[0]}`
                                                : `chunks ${s[0]}\u2013${s[1]}`;
    });
  }

  const LANE = "position:relative;height:%h;background:#15181b;border:1px solid " +
               "#2c3238;border-radius:4px;margin-bottom:5px;";
  const laneLabel = (s) => el("div", { textContent: s, style:
    "font-size:9.5px;letter-spacing:.9px;text-transform:uppercase;color:#69727b;" +
    "margin:5px 0 3px;font-family:ui-monospace,monospace;" });
  const SHOT_COLOURS = ["#3f5a6b", "#4a5b45", "#63504a", "#53496b"];

  function drawBoard(t, d) {
    const total = t.total_frames || 1;
    const pc = (f) => `${Math.max(0, f / total * 100)}%`;
    boardInner.innerHTML = "";

    // ruler
    const ruler = el("div", { style: "position:relative;height:15px;margin-bottom:2px;" });
    for (let s = 0; s <= Math.floor(total / 24); s += 2) {
      const tick = el("div", { textContent: `${s}s`, style:
        "position:absolute;top:0;border-left:1px solid #2c3238;padding-left:3px;" +
        "font-size:9px;color:#69727b;font-family:ui-monospace,monospace;" });
      tick.style.left = pc(s * 24);
      ruler.append(tick);
    }
    boardInner.append(ruler);

    // shots — click to select, drag an edge to resize
    boardInner.append(laneLabel("shots — drag an edge to resize"));
    const shotLane = el("div", { style: LANE.replace("%h", "34px") });
    (t.shots || []).forEach((s, i) => {
      const blk = el("div", { style:
        "position:absolute;top:3px;bottom:3px;border-radius:3px;display:flex;" +
        "align-items:center;gap:6px;padding:0 7px;overflow:hidden;cursor:pointer;" +
        `background:${SHOT_COLOURS[i % 4]};border:1px solid ${i === selShot ? "#7fa0c0" : "rgba(255,255,255,.14)"};` });
      blk.style.left = pc(s.start); blk.style.width = pc(s.frames);
      const shot = d.shots[i] || {};
      blk.append(el("span", { textContent: shot.description || "untitled shot",
        style: "font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;" }));
      blk.append(el("span", { textContent: `${s.frames}f`, style:
        "font-size:9.5px;color:rgba(255,255,255,.6);font-family:ui-monospace,monospace;" }));
      for (const side of ["left", "right"]) {
        const g = el("div", { style:
          `position:absolute;top:0;bottom:0;width:7px;${side}:0;cursor:ew-resize;` });
        g.onmousedown = (e) => resizeShot(e, i, side === "right");
        blk.append(g);
      }
      blk.onclick = () => { selShot = i; refresh(); };
      shotLane.append(blk);
    });
    boardInner.append(shotLane);

    // chunks, with the carried handle hatched — the part that cannot be spoken in
    boardInner.append(laneLabel("chunks the planner will make"));
    const chunkLane = el("div", { style: LANE.replace("%h", "22px") });
    (t.chunks || []).forEach((c) => {
      const blk = el("div", { style:
        "position:absolute;top:0;height:100%;border-right:1px solid #3a4148;" +
        "display:flex;align-items:center;justify-content:center;font-size:9px;" +
        "color:#69727b;font-family:ui-monospace,monospace;" });
      blk.style.left = pc(c.start); blk.style.width = pc(c.frames);
      blk.textContent = String(c.index);
      if (c.pin) {
        const hatch = el("div", { style:
          "position:absolute;left:0;top:0;height:100%;opacity:.5;background:" +
          "repeating-linear-gradient(-45deg,#6b5530 0 4px,transparent 4px 8px);" });
        hatch.style.width = `${Math.min(100, c.pin / c.frames * 100)}%`;
        blk.append(hatch);
      }
      chunkLane.append(blk);
    });
    boardInner.append(chunkLane);

    // lines, at the second they are spoken
    boardInner.append(laneLabel("what is said, and when"));
    const beatLane = el("div", { style: LANE.replace("%h", "46px") });
    (t.lines || []).forEach((l, i) => {
      const bad = !!l.problem;
      const blk = el("div", { style:
        "position:absolute;height:14px;border-radius:2px;padding:0 4px;font-size:9px;" +
        "font-family:ui-monospace,monospace;white-space:nowrap;overflow:hidden;" +
        "cursor:grab;border:1px solid " +
        (bad ? "#7a4444" : l.placed ? "#7fa0c0" : "rgba(255,255,255,.12)") + ";" +
        `background:${bad ? "#4d2f2f" : "#34474f"};color:${bad ? "#e7b8b8" : "#bcd6de"};` });
      blk.style.top = `${3 + (i % 3) * 15}px`;
      blk.style.left = l.start === null ? "4px" : pc(l.start * 24);
      blk.style.maxWidth = pc(Math.max(30, (l.seconds || 1) * 24 * 1.6));
      blk.textContent = `${l.placed ? "\u21e5 " : ""}${l.who}: ${l.line}`;
      blk.title = (l.problem || "drag to place this line; double-click to let it flow");
      blk.onmousedown = (e) => placeLine(e, l);
      blk.ondblclick = () => { releaseLine(l); };
      beatLane.append(blk);
    });
    boardInner.append(beatLane);
  }

  // ---- dragging ---------------------------------------------------------- //
  function docLine(rec) {
    const shot = (build().shots || [])[rec.shot];
    if (!shot) return null;
    for (const ch of shot.chunks || []) {
      for (const l of ch.lines || []) if (l.line === rec.line && l.who === rec.who) return l;
    }
    return null;
  }

  function resizeShot(e, i, right) {
    e.stopPropagation(); e.preventDefault();
    const rect = boardInner.getBoundingClientRect();
    const cards = [...shotBox.children];
    const per = (lastTiming?.total_frames || 345) / rect.width;
    const x0 = e.clientX;
    const start = Number(cards[i]?._frames || 141);
    const move = (ev) => {
      const want = start + (right ? 1 : -1) * (ev.clientX - x0) * per;
      // 17n+5 — the only lengths the model accepts, so the number you draw to is
      // the number that renders
      const snapped = Math.max(5, Math.round((want - 5) / 17) * 17 + 5);
      cards[i]._frames = snapped;
      const f = cards[i]?.querySelector(".h3-frames");
      if (f) f.value = snapped;
      refresh();
    };
    const up = () => { document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up); refresh(); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  function placeLine(e, rec) {
    e.preventDefault();
    const rect = boardInner.getBoundingClientRect();
    const total = lastTiming?.total_frames || 345;
    const shotStart = (lastTiming?.shots?.[rec.shot]?.start) || 0;
    const move = (ev) => {
      const frames = Math.max(0, (ev.clientX - rect.left) / rect.width * total);
      const secs = Math.max(0, (frames - shotStart) / 24);
      const l = docLine(rec);
      if (l) { l.at = Math.round(secs * 10) / 10; writeBack(); refresh(); }
    };
    const up = () => { document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  function releaseLine(rec) {
    const l = docLine(rec);
    if (l) { delete l.at; writeBack(); refresh(); }
  }

  // placed times live in the DOCUMENT, and the form rebuilds the document from
  // the DOM every refresh -- so they are stashed where the form can see them
  // again, keyed by the line's own text.
  function writeBack() {
    placed.clear();
    for (const s of build().shots || []) {
      for (const ch of s.chunks || []) {
        for (const l of ch.lines || []) if (l.at !== undefined) placed.set(l.line, l.at);
      }
    }
  }

  let lastTiming = null;

  async function drawTiming(d) {
    const t = await post("/script/timing", { document: d, ...planFromGraph() });
    if (!t.ok) { timing.textContent = t.error; return; }
    lastTiming = t;
    drawBoard(t, d);
    const bar = t.chunks.map((c, i) =>
      `chunk ${i}: ${c.pin_s ? `${c.pin_s}s pinned, ` : ""}` +
      `speakable ${c.speech_from}\u2013${c.speech_to}s`).join("    ");
    const lines = t.lines.map((l) => {
      const when = l.start === null ? "  —  " : `${String(l.start).padStart(5)}s`;
      return `${when}  ${l.who}: ${l.line}` + (l.problem ? `   \u2190 ${l.problem}` : "");
    });
    timing.innerHTML = "";
    timing.append(el("div", { textContent: bar, style: "color:#89a;margin-bottom:5px;" }));
    for (const [i, text] of lines.entries()) {
      timing.append(el("div", { textContent: text,
        style: t.lines[i].problem ? "color:#d88;" : "color:#9a9;" }));
    }
    for (const p of t.problems) {
      timing.append(el("div", { textContent: p, style: "color:#dc8;margin-top:5px;" }));
    }
    markShots(t);
  }

  // ---- wiring ------------------------------------------------------------ //
  const addPerson = el("button", { textContent: "Add a person", style: BTN });
  addPerson.onclick = () => { castBox.append(castRow({ kind: "character" })); refresh(); };
  const addPlace = el("button", { textContent: "Add a place", style: BTN });
  addPlace.onclick = () => { castBox.append(castRow({ kind: "setting" })); refresh(); };
  const addShot = el("button", { textContent: "Add a shot", style: BTN });
  addShot.onclick = () => { shotBox.append(shotCard()); refresh(); };

  body.append(board,
              heading("PEOPLE AND PLACES"), castBox, row([addPerson, addPlace]),
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
  panel.append(head, body, timing, foot);
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
    for (const shot of doc?.shots || []) {
      for (const ch of shot.chunks || []) {
        for (const l of ch.lines || []) if (l.at !== undefined) placed.set(l.line, l.at);
      }
    }
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
