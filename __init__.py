"""MiniMax-H3 Toolkit — nodes for prompting, masking, cropping and long-form work.

The model already generates; ComfyUI's stock H3 nodes handle that. What this pack
adds is the knowledge around it — prompt format, the frame and audio grids, mask
geometry, and the settings that were measured rather than guessed.

Modules are named for what they hold. Ones with no nodes at all — timing, geometry,
cropplan, avlatent — carry the arithmetic the nodes share, and are kept torch-free
where possible so they can be tested without ComfyUI.
"""

from .audio import (NODE_CLASS_MAPPINGS as _AUDIO_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _AUDIO_NAMES, PRESETS)
from .analysis import (NODE_CLASS_MAPPINGS as _ANALYSIS_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _ANALYSIS_NAMES)
from .budget import (NODE_CLASS_MAPPINGS as _BUDGET_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _BUDGET_NAMES)
from .character import (NODE_CLASS_MAPPINGS as _CHAR_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _CHAR_NAMES)
from .checkpoint import (NODE_CLASS_MAPPINGS as _CKPT_CLASSES,
                         NODE_DISPLAY_NAME_MAPPINGS as _CKPT_NAMES)
from .chunklora import (NODE_CLASS_MAPPINGS as _CL_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _CL_NAMES)
from .chunkrun import (NODE_CLASS_MAPPINGS as _RUN_CLASSES,
                      NODE_DISPLAY_NAME_MAPPINGS as _RUN_NAMES)
from .crop import (NODE_CLASS_MAPPINGS as _CROP_CLASSES,
                   NODE_DISPLAY_NAME_MAPPINGS as _CROP_NAMES)
from .encode import (NODE_CLASS_MAPPINGS as _ENC_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _ENC_NAMES)
from .h3script import (NODE_CLASS_MAPPINGS as _SCRIPT_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _SCRIPT_NAMES)
from .guidestrength import (NODE_CLASS_MAPPINGS as _GS_CLASSES,
                            NODE_DISPLAY_NAME_MAPPINGS as _GS_NAMES)
from .longform import (NODE_CLASS_MAPPINGS as _LF_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _LF_NAMES)
from .mask import (NODE_CLASS_MAPPINGS as _MASK_CLASSES,
                   NODE_DISPLAY_NAME_MAPPINGS as _MASK_NAMES)
from .periodicity import (NODE_CLASS_MAPPINGS as _PER_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _PER_NAMES)
from .semantic import (NODE_CLASS_MAPPINGS as _SEM_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _SEM_NAMES)
from .prompt_lint import (NODE_CLASS_MAPPINGS as _LINT_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _LINT_NAMES)
from .prompt_links import (NODE_CLASS_MAPPINGS as _LINK_CLASSES,
                           NODE_DISPLAY_NAME_MAPPINGS as _LINK_NAMES)
from .prompt_rewriter import (NODE_CLASS_MAPPINGS as _RW_CLASSES,
                              NODE_DISPLAY_NAME_MAPPINGS as _RW_NAMES)
from .prompt_scene import (NODE_CLASS_MAPPINGS as _SCENE_CLASSES,
                           NODE_DISPLAY_NAME_MAPPINGS as _SCENE_NAMES)
from .story import (NODE_CLASS_MAPPINGS as _STORY_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _STORY_NAMES)
from .windowing import (NODE_CLASS_MAPPINGS as _WIN_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _WIN_NAMES)
from .video import (NODE_CLASS_MAPPINGS as _VID_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _VID_NAMES)
# MODEL-AGNOSTIC, and deliberately so: it knows nothing about H3 and its
# category is plain `video`. It lives here because a second repo is a second
# install and a second deploy, not because it belongs to this model.
from .videotrim import (NODE_CLASS_MAPPINGS as _VT_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _VT_NAMES)

_PARTS = (
    (_AUDIO_CLASSES, _AUDIO_NAMES),
    (_ANALYSIS_CLASSES, _ANALYSIS_NAMES),
    (_BUDGET_CLASSES, _BUDGET_NAMES),
    (_CHAR_CLASSES, _CHAR_NAMES),
    (_CROP_CLASSES, _CROP_NAMES),
    (_ENC_CLASSES, _ENC_NAMES),
    (_CKPT_CLASSES, _CKPT_NAMES),
    (_CL_CLASSES, _CL_NAMES),
    (_RUN_CLASSES, _RUN_NAMES),
    (_SCRIPT_CLASSES, _SCRIPT_NAMES),
    (_SEM_CLASSES, _SEM_NAMES),
    (_GS_CLASSES, _GS_NAMES),
    (_LF_CLASSES, _LF_NAMES),
    (_MASK_CLASSES, _MASK_NAMES),
    (_PER_CLASSES, _PER_NAMES),
    (_LINT_CLASSES, _LINT_NAMES),
    (_LINK_CLASSES, _LINK_NAMES),
    (_RW_CLASSES, _RW_NAMES),
    (_SCENE_CLASSES, _SCENE_NAMES),
    (_STORY_CLASSES, _STORY_NAMES),
    (_VID_CLASSES, _VID_NAMES),
    (_VT_CLASSES, _VT_NAMES),
    (_WIN_CLASSES, _WIN_NAMES),
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _classes, _names in _PARTS:
    # a duplicate id means two modules claim the same node and one would silently
    # win, so say which rather than letting dict update paper over it
    _clash = set(_classes) & set(NODE_CLASS_MAPPINGS)
    if _clash:
        raise RuntimeError(f"duplicate node id(s) across modules: {sorted(_clash)}")
    NODE_CLASS_MAPPINGS.update(_classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_names)

# serves the widget-hiding / preset / plan-display extension
WEB_DIRECTORY = "./web"

ROUTE_PREFIX = "/h3_toolkit"


def _register_routes():
    """Let the browser fetch a preset so it can fill the widgets visibly.

    The node also applies presets server-side, so the graph still renders correctly
    if this route is unavailable — this only exists so the fields VISIBLY populate
    and stay editable rather than being silently overridden at run time.
    """
    try:
        from server import PromptServer
        from aiohttp import web
        # `instance` only exists once the server is up — importing this module
        # outside a running ComfyUI (a test, a lint pass) must not explode
        routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
    except Exception:
        return
    if routes is None:
        return

    @routes.get(ROUTE_PREFIX + "/preset")
    async def _preset(request):
        name = request.rel_url.query.get("name", "")
        data = PRESETS.get(name)
        if not data:
            return web.json_response({}, status=404)
        return web.json_response(data)

    # --- the script document, for an editor ------------------------------- #
    #
    # THE DOCUMENT IS THE THING; the DSL is one serialization of it. These three
    # routes are the whole server side of a shot/dialogue editor: text in, a
    # document out, a document in, text back, and a dry-run compile so a panel
    # can show the <Subject n> / <Picture n> numbering and the lint WITHOUT
    # queueing a render.
    #
    # A ScriptError carries the line it is on, so a bad script answers 200 with
    # ok:false rather than a 500 — an editor wants to show the message next to
    # the line, not a stack trace.

    async def _body(request):
        try:
            return await request.json()
        except Exception:
            return {}

    @routes.get("/videotrim/probe")
    async def _videotrim_probe(request):
        """A video file's frame count, rate and size — one call per file.

        THE BROWSER CANNOT WORK OUT A FRAME RATE. HTMLVideoElement knows
        `duration` and nothing else, and getting frames and fps client-side
        means WebCodecs and a demuxer. So the trim timeline asks once when a
        file is chosen, and every drag after that is arithmetic in the page.

        Note what is NOT here: no preview endpoint. VHS re-encodes the trimmed
        range through ffmpeg on every widget change; the panel points a <video>
        at core's own /view route instead and nothing is transcoded.
        """
        name = request.rel_url.query.get("file", "")
        try:
            import folder_paths
            from .videotrim import _probe as probe_file
            path = folder_paths.get_annotated_filepath(name)
            _, frames, fps, w, h = probe_file(path)
            return web.json_response({"ok": True, "frames": frames, "fps": fps,
                                      "width": w, "height": h,
                                      "duration": (frames / fps) if fps else 0.0})
        except Exception as exc:
            return web.json_response({"ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"})

    @routes.get(ROUTE_PREFIX + "/characters")
    async def _characters(request):
        """Saved characters, so the panel offers a list instead of a spelling."""
        try:
            from .character import list_characters
            return web.json_response({"ok": True, "characters": list_characters()})
        except Exception as exc:
            return web.json_response({"ok": False, "characters": [],
                                      "error": f"{type(exc).__name__}: {exc}"})

    @routes.post(ROUTE_PREFIX + "/chunk/plan")
    async def _chunk_plan(request):
        """How long will this be, and how many beats do I write?

        THE SAME ARITHMETIC THE NODE USES, over HTTP, so a front end can show a
        user the answer BEFORE queueing anything. Two implementations of one
        rule is the failure this pack keeps paying for, so both call
        `chunkplan.count_plan` and neither owns a copy.

        POST {"chunk_frames": 141, "chunk_count": 4, "context": 39}
          or {"chunk_seconds": 5, "chunk_count": 4}      (snapped, and told so)
        """
        from .chunkplan import (AV_EXACT_RUNS, LEGAL_RUNS, count_plan,
                                describe_count_plan, total_for_count)
        data = await _body(request)
        try:
            ctx = int(data.get("context", 39))
            n = int(data.get("chunk_count", 4))
            fps = float(data.get("fps", 24.0))
            asked_s = data.get("chunk_seconds")
            if data.get("chunk_frames") is not None:
                cf = int(data["chunk_frames"])
                snapped_from = None
            elif asked_s is not None:
                want = max(1, round(float(asked_s) * fps))
                cf = min(LEGAL_RUNS, key=lambda x: (abs(x - want), x))
                snapped_from = want
            else:
                return web.json_response({"ok": False,
                    "error": "give chunk_frames or chunk_seconds"})
            total, first, rest, notes = count_plan(cf, n, ctx)
            return web.json_response({
                "ok": True,
                "chunk_frames": cf, "chunk_count": n, "context": ctx,
                "total_frames": total, "total_seconds": round(total / fps, 3),
                # the asymmetry is arithmetic and the caller has to render it
                "first_chunk_frames": first,
                "first_chunk_seconds": round(first / fps, 3),
                "other_chunk_frames": rest,
                "other_chunk_seconds": round(rest / fps, 3),
                "beats_required": n,
                "snapped_from_frames": snapped_from,
                "legal_runs": [r for r in LEGAL_RUNS if 39 <= r <= 400],
                "av_exact_runs": [r for r in AV_EXACT_RUNS if 39 <= r <= 400],
                "alternatives": [
                    {"chunk_frames": c, "total_frames": total_for_count(c, n, ctx),
                     "total_seconds": round(total_for_count(c, n, ctx) / fps, 3)}
                    for c in LEGAL_RUNS if 39 <= c <= 400 and c != cf],
                "notes": notes,
                "info": describe_count_plan(cf, n, ctx, fps),
            })
        except (TypeError, ValueError) as exc:
            return web.json_response({"ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"})

    @routes.post(ROUTE_PREFIX + "/script/parse")
    async def _script_parse(request):
        from .h3script import ScriptError, parse
        data = await _body(request)
        try:
            return web.json_response({"ok": True,
                                      "document": parse(data.get("text", ""))})
        except ScriptError as exc:
            return web.json_response({"ok": False, "error": str(exc)})

    @routes.post(ROUTE_PREFIX + "/script/serialize")
    async def _script_serialize(request):
        from .h3script import ScriptError, serialize
        data = await _body(request)
        try:
            return web.json_response({"ok": True,
                                      "text": serialize(data.get("document"))})
        except (ScriptError, KeyError, TypeError) as exc:
            return web.json_response({"ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"})

    @routes.post(ROUTE_PREFIX + "/script/timing")
    async def _script_timing(request):
        """Where the lines land, and which ones the chunking will eat."""
        from .h3script import ScriptError, parse, timing
        data = await _body(request)
        try:
            doc = data.get("document")
            if doc is None:
                doc = parse(data.get("text", ""))
            return web.json_response({"ok": True, **timing(
                doc,
                total_frames=int(data.get("total_frames", 345)),
                chunk_frames=int(data.get("chunk_frames", 141)),
                context=int(data.get("context", 39)),
                mode=data.get("mode", "fixed"))})
        except ScriptError as exc:
            return web.json_response({"ok": False, "error": str(exc)})
        except (KeyError, TypeError, ValueError) as exc:
            return web.json_response({"ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"})

    @routes.post(ROUTE_PREFIX + "/script/compile")
    async def _script_compile(request):
        """Numbering + lint for a document, without running anything."""
        from .h3script import ScriptError, emit, lint, parse, serialize
        data = await _body(request)
        try:
            doc = data.get("document")
            if doc is None:
                doc = parse(data.get("text", ""))
            out = emit(doc)
            return web.json_response({
                "ok": True,
                "document": doc,
                "text": serialize(doc),
                "index": out["index"],
                "counts": out["counts"],
                # the text that actually reaches the model. A tool built to stop
                # five nodes disagreeing should let you read what it produced.
                "fields": {k: v for k, v in out.items()
                           if isinstance(v, (str, int, float))},
                "lint": lint(doc, out),
            })
        except ScriptError as exc:
            return web.json_response({"ok": False, "error": str(exc)})
        except (KeyError, TypeError) as exc:
            return web.json_response({"ok": False,
                                      "error": f"{type(exc).__name__}: {exc}"})


    # ------------------------------------------------------------------ #
    # A TAKE AS ONE REQUEST. Three calls, so a front end never builds a
    # graph: ask what you will get, ask for it, ask whether it is done.
    # ------------------------------------------------------------------ #

    @routes.post(ROUTE_PREFIX + "/take/plan")
    async def _take_plan(request):
        """What will this take be? Cheap enough to call on every keystroke.

        POST {"chunk_frames": 192, "chunk_count": 3, "context": 39,
              "beats": "line one\nline two\nline three"}

        Returns the total, the per-chunk boundaries in seconds, which beat
        lands in which chunk, and the lint. Builds no graph and touches no GPU.
        """
        from .chunkplan import count_plan
        from .takeapi import TakeError, plan_fields
        data = await _body(request)
        try:
            plan, lint = plan_fields(data, count_plan)
            return web.json_response({"ok": True, "lint": lint, **plan})
        except (TakeError, ValueError, TypeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)})

    @routes.post(ROUTE_PREFIX + "/take/graph")
    async def _take_graph(request):
        """Compose a submittable API graph. -> {"ok": true, "graph": {...}}

        Same body as /take/plan plus what a render needs: `models`
        (unet / clip / vae / audio_vae / loras), `references`, the prompt
        fields (head / beats / tail / subject_def_1 / retention_1 /
        soundscape / music), `seed`, `lora_schedule`, `filename_prefix`.

        THIS COMPOSES AND DOES NOT QUEUE, on purpose. Queueing means reproducing
        core's /prompt contract -- validate_prompt's signature, the queue tuple,
        the sensitive-key split, the node-replace pass -- and that contract
        MOVES: the tuple gained a sixth element between the version this pack
        documents and the one it runs on. A front end posts this graph to
        ComfyUI's own /prompt, which is the stable public API, and gets core's
        validation errors in core's format for free. `run_take.py` does exactly
        that if you want one call from a shell.
        """
        from .takeapi import TakeError, build
        data = await _body(request)
        try:
            graph = build(data)
        except (TakeError, ValueError, TypeError, KeyError) as exc:
            return web.json_response({"ok": False, "error": str(exc)})
        return web.json_response({"ok": True, "graph": graph,
                                  "post_to": "/prompt"})

    @routes.get(ROUTE_PREFIX + "/take/status")
    async def _take_status(request):
        """Is it done, and what did it write? -> state + output files.

        GET /h3_toolkit/take/status?id=<prompt_id>

        Core's /history already answers this; what it does not do is dig the
        video out of whichever node happened to save it. This flattens that to
        a list a front end can link straight to /view.
        """
        pid = request.rel_url.query.get("id")
        if not pid:
            return web.json_response({"ok": False,
                                      "error": "pass ?id=<prompt_id>"})
        q = PromptServer.instance.prompt_queue
        try:
            running, pending = q.get_current_queue()
        except Exception:                                   # noqa: BLE001
            running, pending = [], []
        in_flight = any(i[1] == pid for i in running)
        waiting = any(i[1] == pid for i in pending)
        entry = (q.get_history(prompt_id=pid) or {}).get(pid)
        if entry is None:
            state = ("running" if in_flight else
                     "queued" if waiting else "unknown")
            return web.json_response({"ok": True, "prompt_id": pid,
                                      "state": state, "outputs": []})
        files = []
        for out in (entry.get("outputs") or {}).values():
            for key in ("images", "gifs", "video", "videos", "audio"):
                for f in out.get(key) or []:
                    if isinstance(f, dict) and f.get("filename"):
                        files.append({"filename": f["filename"],
                                      "subfolder": f.get("subfolder", ""),
                                      "type": f.get("type", "output")})
        status = entry.get("status") or {}
        return web.json_response({
            "ok": True,
            "prompt_id": pid,
            "state": status.get("status_str") or "done",
            "outputs": files,
            "messages": status.get("messages") or [],
        })


_register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
