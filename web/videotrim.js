// The timeline — the whole reason this node exists.
//
// WHAT IT REPLACES
//   VHS shows you a trim by re-encoding it: every change to skip_first_frames
//   or frame_load_cap posts to /viewvideo, which shells out to ffmpeg, filters
//   the range and streams it back. Correct, and you wait for every twiddle.
//
//   Here the <video> element points at core's own /view route — the original
//   file, decoded by the browser — and the trim is drawn over it. Dragging a
//   handle moves a div and seeks a video. Nothing is transcoded, ever, and the
//   only request to the server is one header probe per file.
//
// THE WIDGET IS A VIEW, THE WIDGETS ARE THE TRUTH
//   Everything this draws comes from the node's own `spans`, `span_index`,
//   `force_rate` and `select_every_nth` widgets, and every edit writes back to
//   them. So the node works with this file deleted, the numbers are visible on
//   the node, and they land in the saved workflow — and therefore in the
//   metadata of every mp4 rendered from it.
//
// EVERY NUMBER HERE IS AN OUTPUT FRAME
//   After force_rate and select_every_nth. See videoframes.py for why that is the
//   only domain in which a frame count is a promise the node can keep. The
//   conversion to video time happens in exactly one place, `outToTime`.
import { app } from "../../scripts/app.js";

// both nodes get the same timeline. The batch node has no `span_index` widget
// — every span is emitted — and the panel already treats that widget as
// optional, so nothing branches on which node it is attached to.
const NODES = new Set(["VideoTrimLoad", "VideoTrimLoadAll"]);
const MIN_SPAN = 1;
// THE PANEL'S HEIGHT IS DERIVED FROM THE NODE'S WIDTH, which is how VHS does
// it and the only arrangement here that cannot feed back on itself:
//
//     height = CHROME + (width - GUTTER) / aspect
//
// Width and aspect are both independent of height, so resizing the node
// vertically changes nothing and there is nothing to ratchet. The trap
// MAINodes documented is a computeSize that reads node.size[1]; this reads
// size[0]. That distinction is the whole design.
const CHROME = 285;                    // every fixed row: upload, banner,
                                       // transport, the 44px ruler, two field
                                       // rows, the hint, the gaps and padding
const GUTTER = 20;                     // the panel's own side padding
const MIN_VIDEO = 120;                 // still readable at its smallest
// AND A CAP, which is where this parts company with VHS. A 9:16 clip in a
// 460-wide node is 818px of picture; add the controls and the node is 1100
// tall and you cannot see what you are trimming. Past the cap the picture
// stops growing and centres in the band instead.
const MAX_VIDEO = 520;
const PANEL_FLOOR = CHROME + MIN_VIDEO;
// CHROME IS ONLY TRUE ABOVE THIS WIDTH. Narrower and the rows wrap -- the
// fields take two lines, the hint takes four -- so the chrome is no longer the
// constant the height is computed from and the panel overflows the node. The
// width is therefore not a suggestion.
const MIN_WIDTH = 460;

// The panel's height for a node of this width showing a clip of this aspect.
// Module level and pure, because the widget and its test must run the SAME
// code: the first version of this lived inside the widget and the test
// reimplemented the formula beside it -- so the test passed while the real one
// threw ReferenceError on a variable that was not in scope there.
function panelHeight(width, aspect) {
  const w0 = Math.max(160, (width || 460) - GUTTER);
  const picture = aspect > 0
    ? Math.min(MAX_VIDEO, Math.max(MIN_VIDEO, w0 / aspect))
    : MIN_VIDEO;
  return CHROME + Math.round(picture);
}

// ---------------------------------------------------------------- helpers --
function el(tag, style, props = {}) {
  const n = Object.assign(document.createElement(tag), props);
  if (style) n.style.cssText = style;
  return n;
}

function fmtTime(s) {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toFixed(2).padStart(5, "0")}`;
}

function widgetOf(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function readNum(node, name, fallback) {
  const v = Number(widgetOf(node, name)?.value);
  return Number.isFinite(v) ? v : fallback;
}

// `start count [label]` per line — the same grammar videoframes.py parses, because
// this file writes what that file reads and a second grammar is a second bug.
function parseSpans(text, total) {
  const out = [];
  for (const raw of (text || "").split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^(\d+)\s+(\d+)\s*(.*)$/);
    if (!m) continue;
    const span = { start: +m[1], count: +m[2], label: m[3].trim() };
    // ONLY CLAMP AGAINST A KNOWN CLIP. clampSpan with a total of 0 returns a
    // zeroed span, so parsing before the probe lands -- or after it fails --
    // would erase what it was asked to read. videoframes.py guards this the same
    // way; the two parse the same grammar and must lose data the same way too,
    // which is to say not at all.
    out.push(total ? clampSpan(span, total) : span);
  }
  if (!out.length) out.push({ start: 0, count: total, label: "" });
  return out;
}

function formatSpans(spans) {
  return spans
    .map((s) => `${s.start | 0} ${s.count | 0}${s.label ? " " + s.label : ""}`)
    .join("\n");
}

// ONE EDIT: the start is honoured because you just typed it, and the count is
// cut to whatever is left.
function clampSpan(s, total) {
  if (!total) return { ...s, start: 0, count: 0 };
  const start = Math.max(0, Math.min(s.start | 0, total - 1));
  let count = s.count | 0;
  if (count <= 0) count = total - start;
  return { ...s, start, count: Math.max(0, Math.min(count, total - start)) };
}

// THE CLIP CHANGED, THE SPAN DID NOT: loading the next file, or changing
// force_rate. Here the LENGTH is what you chose and the position is incidental,
// so keep the length and slide the start back -- the same thing dragging the
// block against the end of the timeline does.
//
// Clamping instead is what produced the stuck-at-1 bug: step onto a clip
// shorter than the span's start and clampSpan pinned the start to the last
// frame, cut the count to 1, and wrote that back -- poisoning every later clip.
function refitSpan(s, total) {
  if (!total) return { ...s };
  let count = s.count | 0;
  if (count <= 0) count = total;
  count = Math.min(count, total);
  return { ...s, start: Math.max(0, Math.min(s.start | 0, total - count)), count };
}

// ------------------------------------------------------------------ panel --
function buildPanel(node) {
  // height:100% AND box-sizing, because a DOM widget is given a box and the
  // column has to FILL it. Without this the panel is only as tall as its
  // content, `flex:1` on the preview has nothing to grow into, and the height
  // has to be computed instead -- which is the mistake every earlier version
  // of this made.
  const root = el("div",
    "display:flex;flex-direction:column;gap:6px;padding:6px 4px;" +
    "height:100%;box-sizing:border-box;min-height:0;" +
    "font:12px system-ui,sans-serif;color:#ddd;outline:none;");
  // focusable, because the keyboard shortcuts must NOT be bound to window:
  // ComfyUI already owns space (pan the canvas) and the arrows (nudge a node),
  // and stealing them globally breaks the graph everywhere else
  root.tabIndex = 0;
  // A POINTERDOWN THAT REACHES THE CANVAS DRAGS THE NODE. Every press in here
  // is the start of a scrub or a handle drag, so the canvas must never see one
  // -- otherwise dragging a trim handle moves the node out from under it.
  //
  // stopPropagation only. NOT preventDefault: the default action is what opens
  // a file picker from a label and what focuses an input, and cancelling it
  // here would disable every control in the panel at once.
  root.addEventListener("pointerdown", (e) => e.stopPropagation());

  // THE PREVIEW TAKES WHAT IS LEFT. The fixed rows are sized by their content
  // and never squeezed; the wrapper grows into everything they do not use, so
  // making the node taller makes the picture bigger. That is what a resizable
  // node should do, and it is why nothing here needs to compute a height.
  //
  // The video keeps its own aspect INSIDE the wrapper -- auto dimensions
  // bounded by 100%, which is how a replaced element scales. Space left over is
  // the wrapper's background, not bars on the picture.
  const videoWrap = el("div",
    "flex:1 1 auto;min-height:" + MIN_VIDEO + "px;display:flex;" +
    "align-items:center;justify-content:center;background:#000;" +
    "border-radius:4px;overflow:hidden;");
  // the wrapper still flexes INSIDE the panel, so when the cap has taken over
  // and the band is taller than the picture, the picture sits centred in it
  const video = el("video", "max-width:100%;max-height:100%;display:block;",
    { preload: "metadata", muted: true, playsInline: true, autoplay: true,
      loop: true });
  video.controls = false;
  video.removeAttribute("controls");
  videoWrap.append(video);

  // THE TWO NODES HAVE THE SAME FACE, and what separates them is invisible:
  // one emits a span, the other emits all of them and makes the graph below run
  // once each. Say so, on the node, rather than in a tooltip nobody opens.
  const banner = el("div",
    "background:#23303a;border:1px solid #35505f;border-radius:4px;" +
    "padding:5px 8px;color:#a8c4d6;font-size:11px;line-height:1.45;");

  const status = el("div",
    "display:flex;gap:10px;align-items:center;color:#9aa3ac;font-size:11px;" +
    "flex-wrap:nowrap;white-space:nowrap;overflow:hidden;min-height:24px;");
  const BTN0 = "background:#3a3a3a;color:#ddd;border:1px solid #555;" +
               "border-radius:4px;padding:2px 9px;cursor:pointer;font-size:12px;";
  const playBtn = el("button", BTN0, { textContent: "▶" });
  // A LOCK, not a toggle of the current state. Audio follows the pointer; this
  // says whether it is allowed to. Off means silent even on hover, which is
  // what you want with six of these on a canvas.
  const muteBtn = el("button", BTN0,
    { textContent: "🔊", title: "audio on hover — click to silence (m)" });
  // A LABEL, NOT A BUTTON CALLING .click(). Forwarding a click to a hidden
  // input depends on the gesture surviving whatever the frontend does to
  // events inside a DOM widget; a <label> wrapping its own input opens the
  // picker natively with no JavaScript in the path at all. One less thing
  // between the press and the file dialog.
  // STYLED AS A COMFYUI WIDGET ROW, because that is what it replaces. The
  // button this stands in for is drawn by LiteGraph as a full-width rounded
  // pill in the widget stack; matching it means the panel does not announce
  // that the upload moved.
  const picker = el("input", "position:absolute;width:0;height:0;opacity:0;",
    { type: "file", accept: "video/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.gif" });
  const openBtn = el("label",
    "position:relative;display:block;box-sizing:border-box;width:100%;" +
    "background:#1e1e1e;color:#ccc;border:1px solid #4a4a4a;" +
    "border-radius:8px;padding:4px 10px;text-align:center;cursor:pointer;" +
    "font-size:12px;line-height:1.5;user-select:none;",
    { textContent: "choose file to upload",
      title: "or drop a video anywhere on this panel" });
  openBtn.append(picker);
  openBtn.addEventListener("pointerenter", () => {
    openBtn.style.background = "#2a2a2a";
  });
  openBtn.addEventListener("pointerleave", () => {
    openBtn.style.background = "#1e1e1e";
  });
  const clock = el("span", "font-variant-numeric:tabular-nums;");
  const counter = el("span",
    "font-variant-numeric:tabular-nums;margin-left:auto;color:#93a7ba;" +
    "overflow:hidden;text-overflow:ellipsis;");

  // the ruler
  // NO overflow:hidden. The handles sit at left:0% and left:100% with a
  // negative margin so they straddle the edge -- and a clipping parent removes
  // the outer half of each, which at a full-clip span leaves almost nothing to
  // grab. That is why the ends would not drag.
  //
  // touch-action:none as well: without it a pointer gesture that starts here
  // can be claimed as a scroll and the drag never gets its moves.
  const bar = el("div",
    "position:relative;height:44px;background:#1b1f23;border-radius:4px;" +
    "cursor:pointer;user-select:none;touch-action:none;");
  const region = el("div",
    "position:absolute;top:0;bottom:0;background:rgba(224,122,60,.28);" +
    "border-top:2px solid #e07a3c;border-bottom:2px solid #e07a3c;cursor:grab;");
  // wider than they look: the visible bar is 5px, the grab area is 15px, so
  // the ends are reachable without pixel-hunting
  const HANDLE = "position:absolute;top:-3px;bottom:-3px;width:15px;" +
                 "margin-left:-7px;cursor:ew-resize;touch-action:none;" +
                 "background:linear-gradient(90deg,transparent 33%,#e07a3c 33%," +
                 "#e07a3c 67%,transparent 67%);z-index:3;";
  const hIn = el("div", HANDLE);
  const hOut = el("div", HANDLE);
  const head = el("div",
    "position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;" +
    "background:#fff;pointer-events:none;box-shadow:0 0 4px rgba(0,0,0,.8);");
  bar.append(region, hIn, hOut, head);

  const fields = el("div", "display:flex;gap:6px;align-items:center;");
  const FIELD = "background:#262626;color:#eee;border:1px solid #444;" +
                "border-radius:4px;padding:3px 6px;font-size:12px;width:74px;";
  const inF = el("input", FIELD, { type: "number", min: "0", step: "1" });
  const cntF = el("input", FIELD, { type: "number", min: "1", step: "1" });
  const labelF = el("input",
    FIELD + "width:auto;flex:1;", { placeholder: "label (optional)" });
  const lab = (t) => el("span", "color:#888;font-size:11px;", { textContent: t });

  const BTN = "background:#3a3a3a;color:#ddd;border:1px solid #555;" +
              "border-radius:4px;padding:3px 8px;cursor:pointer;font-size:11px;";
  const spanSel = el("select", FIELD + "width:auto;");
  const addBtn = el("button", BTN, { textContent: "+ span" });
  const delBtn = el("button", BTN, { textContent: "− span" });

  fields.append(lab("in"), inF, lab("frames"), cntF, labelF);
  const spanRow = el("div", "display:flex;gap:6px;align-items:center;");
  spanRow.append(lab("span"), spanSel, addBtn, delBtn);
  spanRow.title = "Mark as many spans as you like while watching once.";

  const hint = el("div", "color:#6d767f;font-size:10.5px;",
    { textContent: "drag the ends to resize · drag the middle to slide the " +
                   "whole block at its length · space play · shift+← → one " +
                   "frame · [ ] set in/out at the playhead · home/end jump" });

  status.append(playBtn, muteBtn, clock, counter);
  root.append(openBtn, banner, videoWrap, status, bar, fields, spanRow, hint);
  // EVERY ROW BUT THE PREVIEW IS FIXED. Set here rather than in eight style
  // strings: a row that is left flexible gets squeezed by a tall clip, and the
  // one that vanished that way was the trim bar.
  for (const child of root.children) {
    if (child !== videoWrap) child.style.flex = "0 0 auto";
  }
  return { root, banner, video, playBtn, muteBtn, openBtn, picker, clock,
           counter, bar, region, hIn, hOut, head, inF, cntF, labelF, spanSel,
           addBtn, delBtn };
}

// ------------------------------------------------------------------ logic --
function attach(node, batch = false) {
  const ui = buildPanel(node);
  // src: the source file's own frames/fps, from one probe. null until it lands.
  let src = null;
  let spans = [];
  let active = 0;
  let drag = null;
  let raf = 0;
  let widgetCount = -1;
  // which file the panel is actually showing, as against which one the widget
  // says. They come apart on every graph load -- see the watcher in tick().
  let loadedName = null;

  // what each span's length was before this clip, so a truncation can be
  // reported rather than just happening
  let wanted = [];
  let shortened = false;

  const rate = () => {
    const forced = readNum(node, "force_rate", 0);
    return forced > 0 ? forced : (src?.fps || 0);
  };
  const step = () => Math.max(1, readNum(node, "select_every_nth", 1) | 0);
  // SOURCE frames -> OUTPUT frames, the same conversion videoframes.py makes, and
  // the reason the ruler is honest when force_rate is not the source's rate
  const total = () => {
    if (!src?.frames || !src?.fps) return 0;
    const r = rate();
    const n = r > 0 && r !== src.fps
      ? Math.floor((src.frames / src.fps) * r) : src.frames;
    return Math.ceil(n / step());
  };
  // the ONE place output frames become video time
  const outToTime = (i) => (rate() > 0 ? (i * step()) / rate() : 0);
  const timeToOut = (t) => (rate() > 0 ? Math.round((t * rate()) / step()) : 0);

  const span = () => spans[active] || { start: 0, count: 0, label: "" };

  function writeBack() {
    const w = widgetOf(node, "spans");
    if (w) {
      w.value = formatSpans(spans);
      w.callback?.(w.value);
    }
    const si = widgetOf(node, "span_index");
    if (si && si.value !== active) { si.value = active; si.callback?.(active); }
    node.setDirtyCanvas(true, true);
  }

  function draw() {
    const t = total();
    const s = span();
    const pct = (f) => (t > 0 ? Math.max(0, Math.min(100, (f / t) * 100)) : 0);
    ui.region.style.left = pct(s.start) + "%";
    ui.region.style.width = Math.max(0, pct(s.start + s.count) - pct(s.start)) + "%";
    ui.hIn.style.left = pct(s.start) + "%";
    ui.hOut.style.left = pct(s.start + s.count) + "%";
    const cur = timeToOut(ui.video.currentTime || 0);
    ui.head.style.left = pct(cur) + "%";

    const fps = rate() / step();
    const clockText =
      `${fmtTime(ui.video.currentTime || 0)} / ${fmtTime(src?.duration || 0)}`;
    if (ui.clock.textContent !== clockText) ui.clock.textContent = clockText;
    const counterText = t
      ? `frame ${cur} / ${t} @ ${(+fps.toFixed(3))}fps  ·  span ` +
        `${s.start}–${s.start + s.count} = ${s.count} (${fmtTime(s.count / (fps || 1))})`
      : "no video";
    if (ui.counter.textContent !== counterText) {
      ui.counter.textContent = counterText;
    }
    if (document.activeElement !== ui.inF) ui.inF.value = s.start;
    if (document.activeElement !== ui.cntF) ui.cntF.value = s.count;
    if (document.activeElement !== ui.labelF) ui.labelF.value = s.label || "";

    ui.banner.style.borderColor = shortened ? "#7a5a2a" : "#35505f";
    ui.banner.textContent = shortened
      ? "This clip is shorter than the span you had, so it was cut to fit. " +
        "Set the length again if you need it back."
      : batch
      ? `Emits ALL ${spans.length} span${spans.length === 1 ? "" : "s"}. ` +
        `Every node below this runs ${spans.length} time` +
        `${spans.length === 1 ? "" : "s"}, in one queue run.`
      : `Emits ONE span: number ${active + 1} of ${spans.length}` +
        `${s.label ? ` — ${s.label}` : ""}. Pick it below; the others are kept `
        + `with the workflow.`;

    if (ui.spanSel.options.length !== spans.length ||
        ui.spanSel.selectedIndex !== active) {
      ui.spanSel.replaceChildren(...spans.map((sp, i) =>
        el("option", "", { value: String(i),
          textContent: `${i + 1}. ${sp.start}+${sp.count}${sp.label ? " " + sp.label : ""}` })));
      ui.spanSel.selectedIndex = active;
    }
  }

  function setSpan(next) {
    // you have touched it, so the "this clip was too short" note has been read
    shortened = false;
    const was = span();
    const now = clampSpan({ ...was, ...next }, total());
    // A POINTERMOVE THAT DID NOT CROSS A FRAME IS NOT A CHANGE. Dragging fires
    // dozens of these per second and most land on the frame already set, so
    // without this every one of them writes the widget and seeks the video.
    if (now.start === was.start && now.count === was.count &&
        now.label === was.label) return false;
    spans[active] = now;
    writeBack();
    draw();
    return true;
  }

  // SHOW THE EDGE YOU ARE MOVING, paused or not. Dragging a handle against a
  // still frame is guessing: the whole point of the trim is which frame it
  // starts on. While playing the loop already keeps the playhead inside the
  // span, so this only has to cover the paused case.
  function previewEdge(kind) {
    if (!ui.video.paused) return;
    // the OUT point is the last frame INSIDE the span, not the first one after
    // it -- seeking to spanOut() shows you a frame the render will not contain
    seek(kind === "out"
      ? Math.max(spanIn(), spanOut() - 1 / (rate() || 24))
      : spanIn());
  }

  // ---- getting a file in ---------------------------------------------------
  //
  // THE PANEL COVERS THE NODE, so a file dropped on it lands on this div and
  // the canvas never sees it -- which is why dropping on the node did nothing.
  // Rather than try to forward the event back to a handler I do not own, the
  // panel takes the file itself: it is the same endpoint and the same two
  // steps the frontend's own widget performs.
  async function upload(file) {
    if (!file) return;
    ui.counter.textContent = `uploading ${file.name}…`;
    const body = new FormData();
    // the route is called /upload/image and takes any input file; `type`
    // decides which folder, and a loader reads from input
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "false");
    try {
      const r = await fetch("/upload/image", { method: "POST", body });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const d = await r.json();
      const name = d.subfolder ? `${d.subfolder}/${d.name}` : d.name;
      const w = widgetOf(node, "file");
      if (w) {
        // the combo was built at registration and has never heard of this file
        if (Array.isArray(w.options?.values) && !w.options.values.includes(name)) {
          w.options.values.push(name);
          w.options.values.sort();
        }
        w.value = name;
        w.callback?.(name);
      }
      await loadFile();
    } catch (err) {
      ui.counter.textContent = `upload failed: ${err.message}`;
    }
  }

  // the label opens the picker on its own; this only stops the press reaching
  // the canvas, which would otherwise start dragging the node
  ui.openBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  ui.picker.onchange = () => {
    upload(ui.picker.files?.[0]);
    ui.picker.value = "";          // so the same file can be chosen twice
  };
  for (const t of ["dragenter", "dragover"]) {
    ui.root.addEventListener(t, (e) => {
      e.preventDefault();
      e.stopPropagation();
      ui.root.style.outline = "2px dashed #e07a3c";
    });
  }
  for (const t of ["dragleave", "drop"]) {
    ui.root.addEventListener(t, (e) => {
      e.preventDefault();
      e.stopPropagation();
      ui.root.style.outline = "none";
    });
  }
  ui.root.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) upload(f);
  });

  // ---- the source file ---------------------------------------------------
  async function loadFile() {
    const name = widgetOf(node, "file")?.value;
    // recorded BEFORE the await, so the watcher below does not fire again
    // while this one is still in flight
    loadedName = name || "";
    if (!name) { src = null; draw(); return; }
    ui.video.src = `/view?filename=${encodeURIComponent(name)}` +
                   `&type=input&subfolder=`;
    try {
      const r = await fetch(`/videotrim/probe?file=${encodeURIComponent(name)}`);
      const d = await r.json();
      src = d.ok ? d : null;
    } catch { src = null; }
    // the spans were written against whatever file was here before, so REFIT
    // them to this one -- length first. Parsed against 0 so parseSpans does no
    // clamping of its own; the refit below is the only thing that moves them.
    const t = total();
    spans = parseSpans(widgetOf(node, "spans")?.value, 0)
      .map((sp) => refitSpan(sp, t));
    active = Math.max(0, Math.min(readNum(node, "span_index", 0) | 0,
                                  spans.length - 1));
    shortened = t > 0 && spans.some((sp, i) => sp.count < (wanted[i] ?? sp.count));
    wanted = spans.map((sp) => sp.count);
    // NEVER WRITE BACK AN UNKNOWN OR DEGENERATE CLIP. A failed probe leaves
    // total() at 0, and a clip that measures ONE frame is almost always
    // ComfyUI's count-loop bug rather than a one-frame file -- the server
    // corrects that now, and this is the second line, because persisting a
    // refit against a bad total is what made the damage outlive the clip. A
    // real one-frame clip has nothing to trim, so nothing is lost by not
    // saving it.
    if (t > 1) writeBack();
    draw();
  }

  // ---- dragging ----------------------------------------------------------
  // start / end / region, the same three the dataset-prep timeline uses.
  // Dragging the REGION keeps its width and slides it, which is the whole
  // point when a model wants an exact frame count: lock the length, hunt for
  // the moment.
  // -> a frame, or NULL when the question cannot be answered.
  //
  // RETURNING 0 WAS THE BUG. A hidden or collapsed element measures zero, and
  // 0 is a perfectly good frame number -- so every caller acted on it and the
  // span jumped to one end. `null` cannot be mistaken for an answer, and every
  // caller now bails instead.
  //
  // The ratio itself is zoom-proof: getBoundingClientRect and clientX are both
  // in viewport pixels, so a scaled element scales both sides. What was never
  // safe was measuring an element that is not on screen.
  const frameAt = (clientX) => {
    const r = ui.bar.getBoundingClientRect();
    const t = total();
    if (!(r.width > 1) || !t) return null;
    return Math.round(((clientX - r.left) / r.width) * t);
  };

  function beginDrag(kind, ev) {
    ev.preventDefault();
    ev.stopPropagation();
    const at = frameAt(ev.clientX);
    if (at === null) return;            // nothing measurable to drag against
    drag = { kind, at, start: span().start, count: span().count,
             id: ev.pointerId };
    ui.region.style.cursor = "grabbing";
    // CAPTURE, so the release is delivered here even if the pointer has left
    // the element -- or the element has gone. A pointerup that lands somewhere
    // else is a drag that never ends.
    try { ev.target.setPointerCapture?.(ev.pointerId); } catch { /* ignore */ }
    window.addEventListener("pointermove", onDrag);
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);
  }

  function onDrag(ev) {
    if (!drag) return;
    // NO BUTTON DOWN MEANS NO DRAG. This is the self-heal, and it is what was
    // actually wrong: if a pointerup is ever missed -- released outside the
    // window, swallowed while the element was hidden, taken by another
    // handler -- the listeners stay attached and `drag` stays set forever.
    // Every later pointermove then moves the trim, and a trackpad emits those
    // while zooming. Which is exactly the report: the box moves when you zoom.
    if (ev.buttons === 0) { endDrag(); return; }
    // and ignore a second pointer entirely, rather than letting it fight the
    // first over the same span
    if (drag.id !== undefined && ev.pointerId !== undefined &&
        ev.pointerId !== drag.id) return;
    const t = total();
    const now = frameAt(ev.clientX);
    if (now === null) return;           // hidden mid-drag: hold, do not guess
    const delta = now - drag.at;
    let moved;
    if (drag.kind === "in") {
      const start = Math.max(0, Math.min(drag.start + drag.count - MIN_SPAN,
                                         drag.start + delta));
      moved = setSpan({ start, count: drag.start + drag.count - start });
    } else if (drag.kind === "out") {
      const end = Math.max(drag.start + MIN_SPAN,
                           Math.min(t, drag.start + drag.count + delta));
      moved = setSpan({ start: drag.start, count: end - drag.start });
    } else {
      // slide, keeping the width — and stop AT the edges rather than squashing
      const start = Math.max(0, Math.min(t - drag.count, drag.start + delta));
      moved = setSpan({ start, count: drag.count });
    }
    if (moved) previewEdge(drag.kind);
  }

  function endDrag() {
    drag = null;
    ui.region.style.cursor = "grab";
    window.removeEventListener("pointermove", onDrag);
    window.removeEventListener("pointerup", endDrag);
    window.removeEventListener("pointercancel", endDrag);
  }

  ui.hIn.addEventListener("pointerdown", (e) => beginDrag("in", e));
  ui.hOut.addEventListener("pointerdown", (e) => beginDrag("out", e));
  ui.region.addEventListener("pointerdown", (e) => beginDrag("region", e));
  ui.bar.addEventListener("pointerdown", (e) => {
    if (e.target !== ui.bar) return;          // a handle already took it
    const f = frameAt(e.clientX);
    if (f !== null) seek(outToTime(f));
  });

  // ---- transport ---------------------------------------------------------
  function seek(t) {
    if (!isFinite(t)) return;
    ui.video.currentTime = Math.max(0, Math.min(src?.duration || 0, t));
    draw();
  }
  const spanIn = () => outToTime(span().start);
  const spanOut = () => outToTime(span().start + span().count);

  function tick() {
    // LOOP INSIDE THE SPAN, so what you watch is what the node emits. Checked
    // on a frame loop rather than with a timeupdate listener, which fires about
    // four times a second and would overshoot a short span entirely.
    if (!ui.video.paused) {
      if (ui.video.currentTime >= spanOut() - 0.001 ||
          ui.video.currentTime < spanIn() - 0.25) {
        ui.video.currentTime = spanIn();
      }
    }
    // THE WIDGET AND THE PREVIEW COME APART ON EVERY GRAPH LOAD. onNodeCreated
    // runs BEFORE configure() restores the saved widget values, so the panel
    // loads whatever the combo defaults to -- the first file in the input
    // folder -- and configure then sets the real value SILENTLY, without
    // firing the callback that would reload it. Switching tabs and back is the
    // same path, which is where CJ hit it.
    //
    // A string compare per frame, acting only on a change. Loading a file does
    // not change the file widget, so this cannot feed back on itself the way
    // the height watcher did.
    const want = widgetOf(node, "file")?.value ?? "";
    if (loadedName !== null && want !== loadedName) loadFile();

    // the frontend's own preview widget appears only once a file is chosen, so
    // one check at creation misses it. Watching the COUNT is O(1) per frame;
    // scanning the widgets every frame would not be.
    if ((node.widgets?.length || 0) !== widgetCount) {
      widgetCount = node.widgets?.length || 0;
      suppressForeignPlayers(node, node.__vtWidget);
      hideStorageWidgets(node);
    }
    draw();
    raf = requestAnimationFrame(tick);
  }

  function togglePlay() {
    if (ui.video.paused) {
      if (ui.video.currentTime < spanIn() || ui.video.currentTime >= spanOut()) {
        ui.video.currentTime = spanIn();
      }
      // A REJECTED play() IS SILENT: the promise rejects, the frame never
      // moves, and the node reads as broken. Muted playback is always allowed,
      // so force that rather than leaving nothing happening.
      ui.video.play().catch(() => {
        ui.video.muted = true;
        ui.video.play().catch(() => {});
        ui.counter.textContent = "the browser blocked playback — muted; " +
                                 "click the page once";
      });
      ui.playBtn.textContent = "❚❚";
    } else {
      ui.video.pause();
      ui.playBtn.textContent = "▶";
    }
  }

  // `allowAudio` is the lock; `hovering` is the pointer. The element is muted
  // unless both say otherwise, so there is exactly one place the two combine
  // and no way for them to disagree.
  let allowAudio = true;
  let hovering = false;
  function applyAudio() {
    ui.video.muted = !(allowAudio && hovering);
    ui.muteBtn.textContent = allowAudio ? "🔊" : "🔇";
    ui.muteBtn.title = allowAudio
      ? "audio on hover — click to silence (m)"
      : "silenced — click for audio on hover (m)";
  }
  ui.root.addEventListener("pointerenter", () => { hovering = true; applyAudio(); });
  ui.root.addEventListener("pointerleave", () => { hovering = false; applyAudio(); });

  ui.muteBtn.onclick = (e) => {
    e.stopPropagation();
    allowAudio = !allowAudio;
    applyAudio();
  };
  applyAudio();
  ui.playBtn.onclick = (e) => { e.stopPropagation(); togglePlay(); };
  ui.video.addEventListener("pause", () => { ui.playBtn.textContent = "▶"; });
  ui.video.addEventListener("play", () => { ui.playBtn.textContent = "❚❚"; });

  // ---- keyboard, scoped to this widget ------------------------------------
  root_keys(ui, {
    play: togglePlay,
    stepFrame: (d) => seek(ui.video.currentTime + d * (step() / (rate() || 24))),
    seekBy: (d) => seek(ui.video.currentTime + d),
    markIn: () => {
      // no preview here: you set the in point AT the playhead, so the frame on
      // screen is already the answer
      const f = timeToOut(ui.video.currentTime);
      const end = span().start + span().count;
      setSpan({ start: Math.min(f, end - MIN_SPAN),
                count: Math.max(MIN_SPAN, end - f) });
    },
    markOut: () => {
      const f = timeToOut(ui.video.currentTime);
      setSpan({ count: Math.max(MIN_SPAN, f - span().start) });
    },
    home: () => seek(spanIn()),
    end: () => seek(Math.max(spanIn(), spanOut() - 0.05)),
    mute: () => { allowAudio = !allowAudio; applyAudio(); },
  });

  // ---- fields and spans ---------------------------------------------------
  ui.inF.onchange = () => {
    if (setSpan({ start: +ui.inF.value || 0 })) previewEdge("in");
  };
  ui.cntF.onchange = () => {
    if (setSpan({ count: +ui.cntF.value || 0 })) previewEdge("out");
  };
  ui.labelF.onchange = () => setSpan({ label: ui.labelF.value });
  ui.spanSel.onchange = () => {
    active = ui.spanSel.selectedIndex;
    writeBack();
    seek(spanIn());
  };
  ui.addBtn.onclick = (e) => {
    e.stopPropagation();
    const s = span();
    const t = total();
    // the new span starts where the last one ends and keeps its length, which
    // is what marking consecutive beats of one clip actually looks like
    const start = Math.min(t - 1, s.start + s.count);
    spans.push(clampSpan({ start, count: s.count, label: "" }, t));
    active = spans.length - 1;
    writeBack();
    seek(spanIn());
  };
  ui.delBtn.onclick = (e) => {
    e.stopPropagation();
    if (spans.length <= 1) return;
    spans.splice(active, 1);
    active = Math.max(0, active - 1);
    writeBack();
    draw();
  };

  // ---- keep in step with the node's own widgets ---------------------------
  for (const name of ["force_rate", "select_every_nth"]) {
    const w = widgetOf(node, name);
    if (!w) continue;
    const prev = w.callback;
    w.callback = function (...a) {
      const r = prev?.apply(this, a);
      // the ruler's LENGTH just changed, so the spans mean something different.
      // Refit, not clamp: you changed the rate, not the span.
      const t = total();
      spans = spans.map((s) => refitSpan(s, t));
      if (t > 0) writeBack();
      draw();
      return r;
    };
  }
  const fileW = widgetOf(node, "file");
  if (fileW) {
    const prev = fileW.callback;
    fileW.callback = function (...a) {
      const r = prev?.apply(this, a);
      loadFile();
      return r;
    };
  }
  const spansW = widgetOf(node, "spans");
  if (spansW) {
    // hand-edited text is still authoritative; the timeline is the view
    const prev = spansW.callback;
    spansW.callback = function (...a) {
      const r = prev?.apply(this, a);
      spans = parseSpans(spansW.value, total());
      active = Math.min(active, spans.length - 1);
      draw();
      return r;
    };
  }

  ui.video.addEventListener("loadedmetadata", () => {
    // THE ONE MOMENT THE NODE RESIZES: the clip's shape has just become known,
    // and the node's height is a function of it. Everything after this is the
    // user dragging the width, which the frontend recomputes on its own.
    const a = (ui.video.videoWidth || 0) / (ui.video.videoHeight || 1);
    if (a > 0 && a !== node.__vtAspect) {
      node.__vtAspect = a;
      node.setSize?.(node.computeSize());
      node.setDirtyCanvas?.(true, true);
    }
    draw();
  });
  raf = requestAnimationFrame(tick);
  node.onRemoved = ((prev) => function () {
    cancelAnimationFrame(raf);
    ui.video.src = "";
    return prev?.apply(this, arguments);
  })(node.onRemoved);

  loadFile();
  return ui.root;
}

// Bound to the PANEL, never to window. ComfyUI owns space (pan the canvas) and
// the arrow keys (nudge the selected node) on the graph; a global listener here
// would break those everywhere else in the app.
function root_keys(ui, act) {
  ui.root.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement) return;
    let hit = true;
    switch (e.key) {
      case " ": act.play(); break;
      case "ArrowLeft": e.shiftKey ? act.stepFrame(-1) : act.seekBy(-1); break;
      case "ArrowRight": e.shiftKey ? act.stepFrame(1) : act.seekBy(1); break;
      case "[": act.markIn(); break;
      case "]": act.markOut(); break;
      case "Home": act.home(); break;
      case "End": act.end(); break;
      case "m": act.mute(); break;
      default: hit = false;
    }
    if (hit) { e.preventDefault(); e.stopPropagation(); }
  });
}

// The `spans` textarea and `span_index` are the timeline's STORAGE, and raw
// they read as two unlabelled boxes with a number in them. The panel shows the
// same thing as a list you can click, so hide them while it is there.
//
// Hidden BY THE PANEL, deliberately: with this file gone they come back, and
// the node is still fully usable by typing `start count` into a text box. The
// widgets stay the truth; this only stops them being shown twice.
function hideWidget(w) {
  if (!w || w.__vtHidden) return;
  w.__vtHidden = true;
  w.origType = w.type;
  w.origComputeSize = w.computeSize;
  w.type = "hidden";
  w.computeSize = () => [0, -4];        // -4 cancels the row gap
  // A MULTILINE WIDGET IS A DOM ELEMENT, not something the canvas draws. Its
  // `type` only governs the canvas row -- the textarea itself is positioned
  // absolutely over the node and stays exactly where it was, which is how a
  // "hidden" spans box ended up overlapping force_rate.
  const el = w.element || w.inputEl;
  if (el && el.style) el.style.display = "none";
}

function hideStorageWidgets(node) {
  for (const name of ["spans", "span_index"]) {
    hideWidget(node.widgets?.find((x) => x.name === name));
  }
}

// ComfyUI adds its OWN video preview for an upload widget, so the node draws
// two players: ours with the trim on it, and a bare one underneath. Ours is the
// one with the timeline, so the other goes away.
//
// Found by shape rather than by name: whatever the frontend calls its preview
// widget is not ours to depend on, but "a DOM widget holding a <video> that is
// not the one we built" is stable.
function suppressForeignPlayers(node, mine) {
  for (const w of node.widgets || []) {
    if (w === mine || w.__vtHidden) continue;
    const el = w.element || w.inputEl;
    if (!el || el === mine?.element) continue;
    const hasVideo = el.tagName === "VIDEO" ||
                     (el.querySelector && el.querySelector("video"));
    if (hasVideo) hideWidget(w);
  }
}

app.registerExtension({
  name: "VideoTrim.timeline",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.has(nodeData?.name)) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const element = attach(this, nodeData.name === "VideoTrimLoadAll");
      const w = this.__vtWidget = this.addDOMWidget("timeline", "videotrim", element, {
        serialize: false,           // the `spans` widget already carries it
        // KEEP IT MOUNTED. Left to default, the frontend hides a DOM widget
        // when the canvas is zoomed out -- and a hidden element measures zero,
        // which is how zooming alone came to move the trim.
        hideOnZoom: false,
      });
      // NO computeSize. A DOM widget without one is FLEXIBLE: the frontend
      // gives it the node's free space through computeLayoutSize, which reads
      // these two hooks. Without them the widget is allotted nothing and the
      // panel draws straight out through the bottom of the node -- which is
      // exactly what it did.
      if (w) {
        // THE NODE'S HEIGHT FOLLOWS ITS WIDTH. computeSize is handed the
        // width and returns the height the panel needs for a picture of that
        // width at the clip's aspect -- so the video always spans the node and
        // is never letterboxed, and dragging the node wider makes it bigger.
        //
        // Reading size[0] is what makes this safe. A computeSize that reads
        // size[1] describes the height in terms of itself, which is the ratchet
        // MAINodes documented and which I walked into twice by other routes.
        // `self`, not `node`: inside onNodeCreated the node is `this`, and
        // `node` is a parameter of attach() that does not exist in this scope.
        // Captured the way VHS captures previewNode, for the same reason.
        const self = this;
        w.computeSize = (width) =>
          [width, panelHeight(width || self.size?.[0], self.__vtAspect)];
        w.options.getMinHeight = () => PANEL_FLOOR;
        w.options.getMaxHeight = () => 100000;
      }
      // setSize, not `this.size =`: assigning the array skips the frontend's
      // own relayout, so the node keeps its old bounds and the widget is
      // measured against them
      // 16:9 until a clip says otherwise, so a fresh node opens at a sensible
      // shape rather than at its floor. The first loadedmetadata replaces it.
      this.__vtAspect = this.__vtAspect || 16 / 9;
      // TAKE THE HEIGHT, KEEP THE WIDTH. `setSize(this.computeSize())` passes
      // computeSize's own width through as well -- the node's natural minimum,
      // which is narrower than this panel can be laid out at. That is how it
      // ended up thin enough for every row to wrap.
      this.size[0] = Math.max(this.size[0] || 0, MIN_WIDTH);
      this.setSize([this.size[0], this.computeSize()[1]]);
      // and it cannot be dragged back down into wrapping. onResize is given
      // the size being applied and may edit it in place.
      const prevResize = this.onResize;
      this.onResize = function (size) {
        if (size[0] < MIN_WIDTH) size[0] = MIN_WIDTH;
        return prevResize?.apply(this, arguments);
      };
      hideStorageWidgets(this);
      suppressForeignPlayers(this, w);
      return r;
    };
  },
});
