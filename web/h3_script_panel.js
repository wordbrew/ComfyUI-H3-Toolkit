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
// Sentence case, no tracking. A tracked-out ALL-CAPS eyebrow over every section
// is template chrome -- it appears whatever the subject, so it says nothing about
// this one.
function heading(text) {
  return el("div", { textContent: text, style:
    "color:#9aa3ac;font-size:12px;font-weight:600;margin:12px 0 5px;" });
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

  // The two numbers that shape every chunk, on the board rather than buried in a
  // node's widgets — trying a different chunk size is the main thing you would
  // want to DO with a plan you can see. They write back to the node, so the plan
  // it emits is the plan you were looking at.
  const nodeNum = (name, fallback) => {
    const w = node.widgets?.find((x) => x.name === name);
    const v = Number(w?.value);
    return Number.isFinite(v) && v > 0 ? v : fallback;
  };
  const setNodeNum = (name, v) => {
    const w = node.widgets?.find((x) => x.name === name);
    if (w) { w.value = v; w.callback?.(v); }
  };

  const toolbar = el("div", { style:
    "display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px;" +
    "font-size:11.5px;color:#8b949e;" });
  const chunkSel = select(["90", "141", "192", "243", "294", "345"],
                          String(nodeNum("chunk_frames", 141)), "width:78px;");
  const ctxSel = select(["0", "39", "56", "90"],
                        String(nodeNum("context", 39)), "width:70px;");
  const totalOut = el("div", { style:
    "margin-left:auto;font-size:11.5px;color:#93a7ba;" });
  chunkSel.onchange = () => { setNodeNum("chunk_frames", Number(chunkSel.value)); refresh(); };
  ctxSel.onchange = () => { setNodeNum("context", Number(ctxSel.value)); refresh(); };
  toolbar.append(el("span", { textContent: "chunk" }), chunkSel,
                 el("span", { textContent: "carried handle" }), ctxSel, totalOut);

  // THE BOARD. Shots on a real clock, the chunks the planner will make drawn
  // underneath them, and every line at the second it is actually spoken. The
  // point is not tidiness: a carried handle and a chunk boundary are properties
  // of the PLAN, so where a line lands is not guessable while writing it.
  const boardWrap = el("div");
  const board = el("div", { style:
    "background:#1b1f23;border-radius:5px;padding:9px 11px 11px;" +
    "margin-bottom:6px;overflow-x:auto;" });
  const boardInner = el("div", { style: "min-width:640px;position:relative;" });
  board.append(boardInner);
  boardWrap.append(toolbar, board);
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
    } else if (kind === "lora") {
      // a shot's LoRA is a property of the SHOT, and H3 Chunk Lora schedules it
      // by time — so it belongs beside the shot rather than in a separate
      // schedule you keep in step by hand
      const file = input(data?.name, "lora file, as it appears in the picker", "flex:1;");
      const str = input(data?.strength ?? "1.0", "0.8 or 0.4 1.0", "width:96px;");
      file.oninput = refresh; str.oninput = refresh;
      wrap.append(row([el("span", { textContent: "lora",
        style: "color:#888;width:56px;font-size:11px;" }), file, str, del]));
      read = () => ({ type: "lora", name: file.value, strength: str.value });
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
      wrap._frames = Math.max(5, Number(frames.value) || 141);
      refresh();                       // the server snaps; the board reports back
    };
    const del = el("button", { textContent: "Remove shot", style: BTN });
    const beats = el("div", { style: "margin:8px 0 6px;" });

    // a loaded take carries notes, actions, lines and splits; show them in order
    for (const n of shot?.notes || []) beats.append(beatRow("note", n));
    for (const l of shot?.loras || []) beats.append(beatRow("lora", l));
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
    // NO "Split here" any more. A shot's LENGTH decides where the planner cuts,
    // and the board draws the result -- offering a second way to say the same
    // thing invited the two to disagree. Splits in a loaded take are still shown
    // and still parsed, so nothing written before this breaks.
    wrap.append(row([add("Someone speaks", "say"), add("Something happens", "do"),
                     add("A detail to hold", "note"), add("A LoRA for this shot", "lora")],
                    "margin-bottom:0;"));

    wrap._frames = Number(shot?.frames ?? 141);
    wrap._read = () => {
      const out = { description: what.value, notes: [],
                    frames: Number(wrap._frames) || 141,
                    chunks: [{ lines: [], actions: [] }], loras: [] };
      for (const b of beats.children) {
        const v = b._read?.();
        if (!v) continue;
        if (v.type === "note") { if (v.text) out.notes.push(v.text); }
        else if (v.type === "lora") {
          if (v.name) out.loras.push({ name: v.name, strength: v.strength || "1.0" });
        }
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
    const cf = n?.widgets?.find((x) => x.name === "cut_frames");
    return {
      found: !!n,
      cut_frames: (cf?.value ?? "").toString().trim(),
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

  // A TRACK SHEET, not a web page. Every editing tool a ComfyUI user already
  // knows -- Premiere, Resolve, Avid -- puts the track name in a gutter beside
  // its lane, in sentence case, and saves monospace for timecode. Stacking
  // uppercase labels above each lane was my habit, not the subject's idiom.
  const GUTTER = 104;
  const SHOT_COLOURS = ["#3f5a6b", "#4a5b45", "#63504a", "#53496b"];

  // Borders are spent on hierarchy: the shots lane is the thing you manipulate,
  // so it has an edge. The lanes below are DERIVED from it and read as quieter
  // beds — one radius and one border on everything flattens that distinction.
  const BED_OWNED  = "position:relative;height:%h;background:#202428;" +
                     "border:1px solid #333a41;border-radius:3px;";
  const BED_DERIVED = "position:relative;height:%h;background:#1a1d21;" +
                      "border-radius:2px;";

  function lane(name, height, owned) {
    const rowEl = el("div", { style:
      "display:grid;grid-template-columns:" + GUTTER + "px 1fr;gap:8px;" +
      "align-items:start;margin-bottom:4px;" });
    rowEl.append(el("div", { textContent: name, style:
      "font-size:11px;color:#7d868f;text-align:right;padding-top:3px;" +
      "line-height:1.25;" }));
    const bed = el("div", { style: (owned ? BED_OWNED : BED_DERIVED)
                                      .replace("%h", height) });
    rowEl.append(bed);
    boardInner.append(rowEl);
    return bed;
  }

  function drawBoard(t, d) {
    const total = t.total_frames || 1;
    const pc = (f) => `${Math.max(0, f / total * 100)}%`;
    boardInner.innerHTML = "";

    // ruler, aligned to the lane column rather than the whole board
    const ruler = lane("", "15px", false);
    ruler.style.background = "transparent";
    for (let s = 0; s <= Math.floor(total / 24); s += 2) {
      const tick = el("div", { textContent: `${s}s`, style:
        "position:absolute;top:0;border-left:1px solid #2a3036;padding-left:3px;" +
        "font-size:9.5px;color:#6d767f;font-family:ui-monospace,monospace;" +
        "font-variant-numeric:tabular-nums;" });
      tick.style.left = pc(s * 24);
      ruler.append(tick);
    }

    const shotLane = lane("Shots", "34px", true);
    (t.shots || []).forEach((s, i) => {
      const blk = el("div", { style:
        "position:absolute;top:3px;bottom:3px;border-radius:3px;display:flex;" +
        "align-items:center;gap:6px;padding:0 7px;overflow:hidden;cursor:pointer;" +
        `background:${SHOT_COLOURS[i % 4]};border:1px solid ${i === selShot ? "#7fa0c0" : "rgba(255,255,255,.14)"};` });
      blk.style.left = pc(s.start); blk.style.width = pc(s.frames);
      const shot = d.shots[i] || {};
      blk.append(el("span", { textContent: shot.description || "Untitled shot",
        style: "font-size:11.5px;white-space:nowrap;overflow:hidden;" +
               "text-overflow:ellipsis;flex:1;" }));
      const asked = s.requested;
      const label = asked && asked !== s.frames
        ? `${s.frames}f \u2190 ${asked}` : `${s.frames}f`;
      if (asked && asked !== s.frames) {
        blk.title = `${asked} is not a legal run (17n+5); the planner uses ${s.frames}`;
      }
      blk.append(el("span", { textContent: label, style:
        "font-size:10px;color:rgba(255,255,255,.62);font-family:ui-monospace,monospace;" +
        "font-variant-numeric:tabular-nums;" }));
      const f = shotBox.children[i]?.querySelector(".h3-frames");
      if (f && document.activeElement !== f) f.value = s.frames;
      for (const side of ["left", "right"]) {
        const g = el("div", { style:
          `position:absolute;top:0;bottom:0;width:7px;${side}:0;cursor:ew-resize;` });
        g.onmousedown = (e) => resizeShot(e, i, side === "right");
        blk.append(g);
      }
      blk.onclick = () => { selShot = i; refresh(); };
      blk.onmousedown = (e) => {
        if (e.target !== blk) return;           // the grips own the edges
        moveShot(e, i);
      };
      shotLane.append(blk);
    });

    // chunks, with the carried handle hatched — the part that cannot be spoken in
    const chunkLane = lane("Chunks", "22px", false);
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

    // actions, under the shot they belong to. They have no time of their own --
    // H3Dialogue attaches one per chunk clause -- so they are drawn as the span
    // of their shot rather than pretending to a moment.
    const actLane = lane("Action", "20px", false);
    (d.shots || []).forEach((shot, si) => {
      const geom = (t.shots || [])[si];
      if (!geom) return;
      const acts = (shot.chunks || []).flatMap((c) => c.actions || []);
      if (!acts.length) return;
      const blk = el("div", { textContent: `\u25b8 ${acts.join("  \u00b7  ")}`,
        style: "position:absolute;top:3px;height:14px;border-radius:2px;" +
               "padding:0 5px;font-size:9px;font-family:ui-monospace,monospace;" +
               "white-space:nowrap;overflow:hidden;background:#3d3a2c;color:#d6cfae;" +
               "border:1px solid rgba(255,255,255,.12);" });
      blk.style.left = pc(geom.start + 4);
      blk.style.maxWidth = pc(Math.max(30, geom.frames - 8));
      blk.title = acts.join("\n");
      actLane.append(blk);
    });
    boardInner.append(actLane);

    // LoRAs, across the span of the shot they belong to. H3 Chunk Lora addresses
    // them by time on the FINISHED clip, which is not the source time — the join
    // drops each chunk's handle. Seeing the span is the point.
    const anyLora = (d.shots || []).some((s) => (s.loras || []).length);
    if (anyLora) {
      const loraLane = lane("LoRAs", "20px", false);
      (d.shots || []).forEach((shot, si) => {
        const geom = (t.shots || [])[si];
        if (!geom || !(shot.loras || []).length) return;
        const label = shot.loras
          .map((l) => `${(l.name || "").split(/[\\/]/).pop()} ${l.strength}`)
          .join("  \u00b7  ");
        const blk = el("div", { textContent: label, style:
          "position:absolute;top:3px;height:14px;border-radius:2px;padding:0 5px;" +
          "font-size:9px;font-family:ui-monospace,monospace;white-space:nowrap;" +
          "overflow:hidden;background:#3a3050;color:#cdc0e0;" +
          "border:1px solid rgba(255,255,255,.12);" });
        blk.style.left = pc(geom.start + 4);
        blk.style.maxWidth = pc(Math.max(30, geom.frames - 8));
        blk.title = label;
        loraLane.append(blk);
      });
    }

    // lines, at the second they are spoken
    const beatLane = lane("Dialogue", "46px", false);
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
    snapshot();
    const rect = boardInner.getBoundingClientRect();
    const cards = [...shotBox.children];
    const per = (lastTiming?.total_frames || 345) / rect.width;
    const x0 = e.clientX;
    const start = Number(cards[i]?._frames || 141);
    const move = (ev) => {
      // NO SNAPPING HERE. 17n+5 has exactly one implementation -- chunkplan's --
      // and a copy in JavaScript would agree today with nothing keeping it
      // agreeing tomorrow. The raw number goes to the server, which snaps it and
      // returns what will actually render; drawBoard writes that back.
      const want = Math.max(5, Math.round(start + (right ? 1 : -1) * (ev.clientX - x0) * per));
      cards[i]._frames = want;
      const f = cards[i]?.querySelector(".h3-frames");
      if (f) f.value = want;
      refresh();
    };
    const up = () => { document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up); refresh(); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  // REORDER BY WHERE THE MIDDLE ENDS UP, not by how far the pointer travelled,
  // so a short shot can pass a long one without having to cross its whole width.
  function moveShot(e, i) {
    e.preventDefault();
    snapshot();
    const rect = boardInner.getBoundingClientRect();
    const total = lastTiming?.total_frames || 345;
    const per = total / rect.width;
    const x0 = e.clientX;
    let moved = false;
    const move = (ev) => {
      const cards = [...shotBox.children];
      const lens = cards.map((c) => Number(c._frames) || 141);
      const before = lens.slice(0, i).reduce((a, b) => a + b, 0);
      const mid = before + (ev.clientX - x0) * per + lens[i] / 2;
      let acc = 0, to = i;
      for (let k = 0; k < lens.length; k++) {
        if (k === i) { acc += lens[k]; continue; }
        if (mid < acc + lens[k] / 2) { to = k > i ? k - 1 : k; break; }
        acc += lens[k]; to = k;
      }
      if (to !== i && to >= 0 && to < cards.length) {
        const node = cards[i];
        shotBox.removeChild(node);
        shotBox.insertBefore(node, shotBox.children[to] || null);
        i = to; moved = true; selShot = to; refresh();
      }
    };
    const up = () => { document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up); if (moved) refresh(); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  // the DOM row holding one line, so a line can be moved between shots rather
  // than only within one
  function beatNodeFor(rec) {
    for (const card of shotBox.children) {
      for (const r of card.querySelectorAll(":scope > div")) {
        const v = r._read?.();
        if (v && v.type === "say" && v.line === rec.line && v.who === rec.who) {
          return { row: r, card };
        }
      }
    }
    return null;
  }

  function placeLine(e, rec) {
    e.preventDefault();
    const rect = boardInner.getBoundingClientRect();
    const total = lastTiming?.total_frames || 345;
    snapshot();
    const move = (ev) => {
      const frames = Math.max(0, (ev.clientX - rect.left) / rect.width * total);

      // WHICH SHOT IS IT OVER NOW? Dragging a line past its shot's edge used to
      // leave it in the old shot with a time beyond that shot's end, which the
      // timing call then reported as "runs past the end of its chunk" — a
      // complaint about the tool's own behaviour rather than the take.
      const shots = lastTiming?.shots || [];
      let si = shots.findIndex((s) => frames >= s.start && frames < s.start + s.frames);
      if (si < 0) si = frames < 0 ? 0 : shots.length - 1;

      if (si !== rec.shot) {
        const found = beatNodeFor(rec);
        const target = shotBox.children[si];
        if (found && target && found.card !== target) {
          const beats = target.querySelector(":scope > div:nth-of-type(2)")
                        || target;
          beats.append(found.row);
          rec.shot = si;
        }
      }
      const secs = Math.max(0, (frames - (shots[si]?.start || 0)) / 24);
      const l = docLine(rec);
      if (l) { l.at = Math.round(secs * 10) / 10; writeBack(); }
      refresh();
    };
    const up = () => { document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  function releaseLine(rec) {
    snapshot();
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

  // UNDO. Everything on the board is a drag, and a drag has no natural way back
  // — resize a shot by accident and the plan underneath it changes. The document
  // is plain data, so a snapshot is a copy and an undo is a reload.
  const history = [];
  function snapshot() {
    try {
      history.push(JSON.stringify(build()));
      if (history.length > 50) history.shift();
    } catch { /* a half-built form is not worth refusing the gesture over */ }
  }
  function loadDoc(d) {
    placed.clear();
    for (const s of d.shots || []) {
      for (const ch of s.chunks || []) {
        for (const l of ch.lines || []) if (l.at !== undefined) placed.set(l.line, l.at);
      }
    }
    castBox.innerHTML = ""; shotBox.innerHTML = "";
    for (const c of d.cast || []) castBox.append(castRow(c));
    for (const s of d.shots || []) shotBox.append(shotCard(s));
    refresh();
  }
  function undo() {
    const prev = history.pop();
    if (prev) loadDoc(JSON.parse(prev));
  }
  back.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault(); e.stopPropagation(); undo();
    }
    if (e.key === "Escape") { e.stopPropagation(); back.remove(); }
  });

  async function drawTiming(d) {
    const t = await post("/script/timing", {
      document: d,
      chunk_frames: Number(chunkSel.value),
      context: Number(ctxSel.value),
    });
    if (!t.ok) { timing.textContent = t.error; return; }
    lastTiming = t;
    // 'A · B · C' reads as template chrome. Say it.
    const nShots = (d.shots || []).length;
    totalOut.textContent =
      `${nShots} shot${nShots === 1 ? "" : "s"} in ${t.chunks.length} chunks, ` +
      `${(t.total_frames / 24).toFixed(1)}s`;
    drawBoard(t, d);
    const bar = t.chunks.map((c, i) =>
      `${i}  ${c.pin_s ? `${c.pin_s}s held` : "free"}  ` +
      `speak ${c.speech_from}\u2013${c.speech_to}s`).join("     ");
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

    // THE BOARD MUST NOT LIE. It plans from the shot lengths drawn here; the
    // render plans from whatever H3 Chunk Plan holds. Wire this node's `plan`
    // output (or cut_frames/total_frames) into the chain and they are the same
    // object. Until then, say plainly that they might not be.
    const g = planFromGraph();
    const mismatch = [];
    if (g.found) {
      if (g.total_frames !== t.total_frames)
        mismatch.push(`length: this take is ${t.total_frames}f, H3 Chunk Plan says ${g.total_frames}f`);
      if ((g.cut_frames || "") !== (t.cuts || []).join(","))
        mismatch.push(`cuts: this take cuts at ${(t.cuts || []).join(", ") || "none"}, ` +
                      `H3 Chunk Plan has ${g.cut_frames || "none"}`);
    }
    if (mismatch.length) {
      timing.append(el("div", { style: "color:#d97a7a;margin-top:7px;",
        textContent: "the board and the render disagree — " + mismatch.join("; ") +
          ". Wire this node's plan output into the chain, or copy the numbers across." }));
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

  body.append(boardWrap,
              heading("People and places"), castBox, row([addPerson, addPlace]),
              heading("Shots"), shotBox, row([addShot]));

  const showPrompt = el("button", { textContent: "Show the prompt", style: BTN });
  showPrompt.onclick = async () => {
    const r = await post("/script/compile", { document: build() });
    if (!r.ok) { status.textContent = r.error; return; }
    const f = r.fields || {};
    const pane = el("div", { style:
      "position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:10001;display:flex;" +
      "align-items:center;justify-content:center;" });
    const box = el("div", { style:
      "background:#1e1e1e;border:1px solid #555;border-radius:8px;max-width:min(760px,92vw);" +
      "max-height:84vh;overflow:auto;padding:16px;color:#ddd;" });
    box.append(el("div", { textContent: "What reaches the model",
      style: "font-size:15px;font-weight:600;margin-bottom:4px;" }));
    box.append(el("div", { textContent:
      "Each block below is one output on the node.", style:
      "font-size:11.5px;color:#7d868f;margin-bottom:6px;" }));
    for (const key of ["head", "subject_defs", "retention", "soundscape",
                       "music", "dialogue_lines", "dialogue_actions",
                       "speaker_map"]) {
      if (!f[key]) continue;
      // these are the node's OUTPUT NAMES, so they are shown exactly as the
      // sockets spell them — uppercasing `subject_defs` would stop it matching
      // the thing you are meant to find on the node
      box.append(el("div", { textContent: key, style:
        "font-family:ui-monospace,monospace;font-size:11px;color:#7d868f;" +
        "margin:13px 0 3px;" }));
      box.append(el("pre", { textContent: String(f[key]), style:
        "margin:0;white-space:pre-wrap;font-family:ui-monospace,monospace;" +
        "font-size:11.5px;color:#b9c2cb;line-height:1.5;" }));
    }
    pane.append(box);
    pane.onclick = (e) => { if (e.target === pane) pane.remove(); };
    document.body.append(pane);
  };

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
  foot.append(status, showPrompt, cancel, save);
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
