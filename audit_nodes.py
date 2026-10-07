#!/usr/bin/env python3
"""Audit the node surface: dead inputs, unwired outputs, stale docs, silent no-ops.

Sibling to `normalize_workflows.py` (which checks saved graphs against node
declarations) and `validate_workflows.py` (which checks graphs are structurally
sound). This one checks the DECLARATIONS against the CODE, and the code against
the shipped workflows.

    python3 audit_nodes.py            # all checks
    python3 audit_nodes.py --checks inputs,docs

WHAT EACH CHECK IS FOR, and the bug that motivated it:

  inputs   A declared input the entry point cannot receive, or receives and never
           reads. An input that does nothing is worse than a missing one: the
           widget sits there looking operative.
  gated    An input only read under some OTHER input's setting. `cut_frames` was
           ignored whenever chunk_mode was `fixed` -- silently, while the report
           printed "cuts detected at: 612, 1224" -- and cost a 59-second render.
           NOTE this check only sees gating inside the entry function; the
           cut_frames case lived a level down in chunkplan.plan() and this check
           does NOT catch it. Treat a clean result as weak evidence.
  outputs  Outputs never wired in any shipped workflow, split into pipeline types
           (a dead MODEL or CONDITIONING means the node is inert) and display
           types (an unwired `info` is this pack's normal display surface).
           MATCHED BY SLOT INDEX: a saved graph may label slot 0 "MODEL" where
           the node declares "model", and matching on the label reports a wired
           output as dead. It did, on H3ChunkLora.
  labels   Saved output labels that disagree with RETURN_NAMES. Harmless, since
           slots are positional, but it is what makes `outputs` lie if you match
           by name.
  docs     A tooltip or DESCRIPTION naming an input, output or node that does not
           exist. A stale name sends you hunting for a widget that was renamed.
  bare     Inputs with no tooltip at all. This pack's whole value proposition is
           the knowledge around the model, so an undocumented widget is a gap in
           the product, not just in the comments. Nodes marked DEPRECATED are
           skipped: that flag hides them from the add-node menu, so nobody can
           hover a widget that is never placed by hand.
  effects  Side effects in INPUT_TYPES. ComfyUI calls it on every /object_info,
           so a filesystem write there runs on every page load. H3Character's
           character list did an os.makedirs.

Exit code is 0 unless a check in --strict reports something, so CI can gate on
`inputs,docs,labels` while `bare` stays advisory.
"""
import argparse, ast, collections, glob, inspect, json, os, pathlib, sys, textwrap, types

ROOT = pathlib.Path(__file__).resolve().parent
PIPE = {"MODEL", "CONDITIONING", "LATENT", "IMAGE", "MASK", "AUDIO", "VAE",
        "SIGMAS", "GUIDER", "SAMPLER", "NOISE", "CLIP", "VIDEO"}
# real concepts that appear in `backticks` without being input names
DOC_ALLOW = {
    "denoise", "sigma", "sigmas", "steps", "shift", "seed", "latent", "mask",
    "audio", "fps", "frames", "width", "height", "length", "model", "images",
    "image", "prompt", "video", "true", "false", "none", "int", "str", "float",
    "bool", "list", "dict", "tuple", "self", "scene", "fixed", "nearest",
    "down", "up", "head", "tail", "both", "all", "any", "chunks", "chunk",
    "link", "links", "beats", "beat", "shot", "shots", "cut", "cuts", "token",
    "tokens", "run", "runs", "grid", "clock", "clocks", "pin", "carry",
    "context", "start", "end", "keep", "trim", "skip", "pad", "crop", "uncrop",
    "encode", "decode", "vae", "dit", "lora", "loras", "adaln", "rope", "cfg",
    "eta", "euler", "beta", "simple", "normal", "res_multistep", "er_sde",
    "sgm_uniform", "beta57", "match", "max", "min", "stretch", "fill", "cover",
    "contain", "ref", "refs", "reference", "references", "everything",
    # H3Script's own script language, which its tooltip has to name
    "note", "say", "shot", "style", "soundscape", "character", "setting",
    # H3ChunkLora's schedule keywords -- `last` is a chunk reference, not an input
    "last", "end",
    # the continuity vocabulary: transition names, not input names
    "reference_sample", "reference_video", "audio_carry", "audio_reference",
    "handoff", "refresh",
}
# inputs whose meaning is carried by their type
BARE_OK = {"model", "images", "image", "mask", "latent", "audio", "vae", "clip",
           "conditioning", "seed", "plan", "flow"}
HIDDEN = {"unique_id", "prompt", "extra_pnginfo", "node_id", "dynprompt"}


# --------------------------------------------------------------- stub ComfyUI
def load_pack():
    """Import every node module with ComfyUI stubbed. Returns (classes, failures)."""
    def stub(name, **attrs):
        m = sys.modules.setdefault(name, types.ModuleType(name))
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    for n in ("comfy", "comfy.utils", "comfy.model_management", "comfy.samplers",
              "comfy.sample", "comfy.sd", "comfy.ldm", "comfy.cli_args",
              "folder_paths", "torch", "torch.nn", "torch.nn.functional",
              "torchaudio", "numpy", "scipy", "scipy.ndimage", "PIL",
              "PIL.Image", "cv2", "av", "comfy_execution",
              "comfy_execution.graph_utils", "comfy_api", "comfy_api.input",
              "comfy_api.util", "server", "aiohttp", "aiohttp.web", "nodes",
              "safetensors", "safetensors.torch", "node_helpers"):
        stub(n)
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    sys.modules["torch.nn"].Module = object
    sys.modules["comfy_execution.graph_utils"].GraphBuilder = object
    sys.modules["comfy_execution.graph_utils"].is_link = \
        lambda v: isinstance(v, list)
    fp = sys.modules["folder_paths"]
    fp.get_filename_list = lambda *a, **k: []
    fp.get_full_path = lambda *a, **k: None
    fp.get_output_directory = lambda *a, **k: "/tmp"
    fp.get_input_directory = lambda *a, **k: "/tmp"
    fp.models_dir = "/tmp/models"
    fp.folder_names_and_paths = {}
    sys.modules["comfy.cli_args"].args = types.SimpleNamespace()
    sys.modules["node_helpers"].conditioning_set_values = lambda c, v, **k: c
    sys.modules["numpy"].ndarray = object
    sys.modules["numpy"].float32 = float

    class Any:
        """Any io.* symbol: callable, subscriptable, and a usable base class."""
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return Any()
        def __getitem__(self, i): return self
        def __getattr__(self, n): return Any()
        def __iter__(self): return iter(())

    class ComfyNode:
        pass

    class IOMeta(type):
        def __getattr__(cls, k): return Any()

    class IO(metaclass=IOMeta):
        pass
    IO.ComfyNode = ComfyNode        # a class body cannot see function locals
    stub("comfy_api.latest", io=IO, ComfyExtension=object)
    stub("comfy_api.latest.io", ComfyNode=ComfyNode)

    pkg = types.ModuleType("_h3audit")
    pkg.__path__ = [str(ROOT)]
    sys.modules["_h3audit"] = pkg
    skip = {"normalize_workflows", "validate_workflows", "audit_nodes",
            "_avstub", "__init__"}
    mods = sorted(p.stem for p in ROOT.glob("*.py")
                  if not p.stem.startswith("test_") and p.stem not in skip)
    for helper in ("timing", "geometry", "cropplan", "avlatent", "chunkplan",
                   "prompt_lint", "videoframes", "periodicity"):
        if helper in mods:
            mods.remove(helper)
            mods.insert(0, helper)
    classes, failures = {}, []
    import importlib.util
    for n in mods:
        try:
            sp = importlib.util.spec_from_file_location(f"_h3audit.{n}",
                                                        ROOT / f"{n}.py")
            m = importlib.util.module_from_spec(sp)
            sys.modules[f"_h3audit.{n}"] = m
            sp.loader.exec_module(m)
            classes.update(getattr(m, "NODE_CLASS_MAPPINGS", {}) or {})
        except Exception as e:
            failures.append((n, f"{type(e).__name__}: {e}"))
    return classes, failures


def spec_of(cls):
    try:
        return cls.INPUT_TYPES()
    except AttributeError:
        return None                      # a V3 io.Schema node
    except Exception as e:
        return {"__error__": str(e)}


def declared(spec):
    out = {}
    for grp in ("required", "optional", "hidden"):
        for k, v in (spec.get(grp) or {}).items():
            cfg = v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else {}
            out[k] = (grp, v[0] if isinstance(v, (list, tuple)) else v,
                      cfg if isinstance(cfg, dict) else {})
    return out


def entry(cls):
    nm = getattr(cls, "FUNCTION", None)
    return getattr(cls, nm, None) if nm else None


def fn_tree(fn):
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    return next((n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)


# --------------------------------------------------------------------- checks
def check_inputs(classes, _wf):
    out = []
    for name, cls in sorted(classes.items()):
        spec = spec_of(cls)
        if spec is None:
            continue
        if "__error__" in spec:
            out.append((name, f"INPUT_TYPES raises: {spec['__error__'][:70]}"))
            continue
        dec = declared(spec)
        fn = entry(cls)
        if fn is None:
            out.append((name, f"FUNCTION {getattr(cls,'FUNCTION',None)!r} missing"))
            continue
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.name != "self"]
        pnames = {p.name for p in params}
        kw = any(p.kind is p.VAR_KEYWORD for p in params)
        for k in dec:
            if k not in pnames and not kw and k not in HIDDEN:
                out.append((name, f"`{k}` declared but is not a parameter"))
        for p in params:
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.name not in dec:
                out.append((name, f"`{p.name}` is a parameter nothing declares "
                                  f"(always {p.default!r})"))
        node = fn_tree(fn)
        if node:
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            star = "locals" in used
            # A node may legitimately take an input its own entry point never
            # reads -- H3ChunkOpen's resume_* are read by H3ChunkClose, out of
            # the dynprompt, as LINKS rather than values. A class declaring
            # AUDIT_UNREAD_OK says so on purpose, and the reason belongs in a
            # comment beside it.
            ok = set(getattr(cls, "AUDIT_UNREAD_OK", ()) or ())
            if not star:
                for k in dec:
                    if k in pnames and k not in used and k not in ok:
                        out.append((name, f"`{k}` is received and never read"))
            for k in ok:
                if k not in dec:
                    out.append((name, f"AUDIT_UNREAD_OK names `{k}`, which is "
                                      f"not an input"))
        rt = getattr(cls, "RETURN_TYPES", ())
        rn = getattr(cls, "RETURN_NAMES", None)
        if rn is not None and len(rn) != len(rt):
            out.append((name, f"RETURN_NAMES has {len(rn)} for {len(rt)} types"))
    return out


def check_gated(classes, _wf):
    out = []
    for name, cls in sorted(classes.items()):
        spec = spec_of(cls)
        if not spec or "__error__" in spec:
            continue
        dec = set(declared(spec))
        fn = entry(cls)
        node = fn_tree(fn) if fn else None
        if not node:
            continue
        pnames = {a.arg for a in node.args.args if a.arg != "self"} | \
                 {a.arg for a in node.args.kwonlyargs}
        uses = {p: [] for p in pnames}

        def walk(n, stack):
            for child in ast.iter_child_nodes(n):
                if isinstance(child, ast.If):
                    test = ast.unparse(child.test)
                    names = {x.id for x in ast.walk(child.test)
                             if isinstance(x, ast.Name)} & pnames
                    for x in ast.walk(child.test):
                        if isinstance(x, ast.Name) and x.id in uses:
                            uses[x.id].append(stack)
                    for b in child.body:
                        walk(b, stack + [(test, names)])
                    for b in child.orelse:
                        walk(b, stack + [("NOT " + test, names)])
                else:
                    for x in ast.walk(child):
                        if isinstance(x, ast.Name) and x.id in uses:
                            uses[x.id].append(stack)
        walk(node, [])
        for p, stacks in sorted(uses.items()):
            if p not in dec or not stacks or any(len(s) == 0 for s in stacks):
                continue
            common = [g for g in stacks[0] if g[1] - {p}]
            if common and all(any(g in s for g in common) for s in stacks):
                out.append((name, f"`{p}` only when: {common[0][0][:80]}"))
    return out


def check_outputs(classes, wf):
    wired = collections.defaultdict(set)
    seen = collections.Counter()
    for g in wf.values():
        by = {n["id"]: n for n in g.get("nodes", [])}
        for n in g.get("nodes", []):
            seen[n.get("type")] += 1
        for l in g.get("links", []):
            if len(l) >= 6 and l[1] in by:
                wired[by[l[1]]["type"]].add(l[2])          # BY INDEX, not name
    out = []
    for name, cls in sorted(classes.items()):
        if name not in seen:
            continue
        rt = list(getattr(cls, "RETURN_TYPES", ()) or ())
        rn = list(getattr(cls, "RETURN_NAMES", ()) or rt)
        never = [(i, rn[i] if i < len(rn) else f"slot{i}", rt[i])
                 for i in range(len(rt)) if i not in wired[name]]
        for i, nm, ty in never:
            tag = "PIPELINE" if ty in PIPE else "display"
            out.append((name, f"[{tag}] slot {i} `{nm}` ({ty}) wired in none of "
                              f"{seen[name]} use(s)"))
    return out


def check_labels(classes, wf):
    agg = collections.Counter()
    for g in wf.values():
        for n in g.get("nodes", []):
            cls = classes.get(n.get("type"))
            if not cls:
                continue
            rn = list(getattr(cls, "RETURN_NAMES", ()) or
                      getattr(cls, "RETURN_TYPES", ()) or ())
            for i, o in enumerate(n.get("outputs") or []):
                if i < len(rn) and o.get("name") != rn[i]:
                    agg[(n["type"], i, o.get("name"), rn[i])] += 1
    return [(t, f"slot {i}: graph says {got!r}, node declares {want!r} "
                f"({c} file(s))") for (t, i, got, want), c in sorted(agg.items())]


def check_docs(classes, _wf):
    import re
    ident = re.compile(r"`([a-z][a-z0-9_]{2,})`")
    noderef = re.compile(r"`(H3[A-Za-z0-9]+)`")
    every_in, every_out = set(), set()
    for name, cls in classes.items():
        spec = spec_of(cls)
        if spec and "__error__" not in spec:
            every_in |= set(declared(spec))
        every_out |= {str(o) for o in (getattr(cls, "RETURN_NAMES", None) or
                                       getattr(cls, "RETURN_TYPES", ()) or ())}
    # H3's six prompt sections are a real vocabulary that tooltips cite by name,
    # and they are not inputs of anything.
    sections = {"subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape", "non_diegetic_music",
                "integrated_multimodal_description"}
    out = []
    for name, cls in sorted(classes.items()):
        spec = spec_of(cls)
        if not spec or "__error__" in spec:
            continue
        dec = declared(spec)
        own = set(dec) | {str(o) for o in (getattr(cls, "RETURN_NAMES", None) or
                                           getattr(cls, "RETURN_TYPES", ()) or ())}
        # A node's own COMBO OPTIONS are legitimate things for its tooltips to
        # name -- `pyramid`, `looped_uniform`, `per_token`. Without this the check
        # fires on every tooltip that explains the choices it offers, which is the
        # most useful kind of tooltip there is.
        for _g, t, _c in dec.values():
            if isinstance(t, (list, tuple)):
                own |= {str(o).split()[0] for o in t if isinstance(o, str)}
        own |= sections
        texts = []
        if getattr(cls, "DESCRIPTION", None):
            texts.append(("DESCRIPTION", cls.DESCRIPTION))
        for k, (_, _, cfg) in dec.items():
            if cfg.get("tooltip"):
                texts.append((k, cfg["tooltip"]))
        for where, text in texts:
            for m in sorted(set(ident.findall(text))):
                if m in own or m in DOC_ALLOW or m in every_in or m in every_out:
                    continue
                out.append((name, f"{where} names `{m}`, which exists nowhere"))
            for m in sorted(set(noderef.findall(text))):
                if m not in classes:
                    out.append((name, f"{where} names node `{m}`, which does "
                                      f"not exist"))
    return out


def check_bare(classes, _wf):
    out = []
    for name, cls in sorted(classes.items()):
        # DEPRECATED keeps a node OUT of the add-node menu, so nobody places one
        # and nobody can hover its widgets. H3ChunkSlice is internal to
        # H3ChunkClose and exists only to be cloned per chunk; tooltips there
        # would be comments with extra steps, and the class docstring is the
        # right place for what it does.
        if getattr(cls, "DEPRECATED", False):
            continue
        spec = spec_of(cls)
        if not spec or "__error__" in spec:
            continue
        dec = declared(spec)
        bare = [k for k, (grp, _, cfg) in dec.items()
                if grp != "hidden" and not cfg.get("tooltip")
                and k not in BARE_OK and k not in HIDDEN]
        if bare:
            out.append((name, f"{len(bare)} undocumented: {', '.join(bare)}"))
    return out


def check_effects(classes, _wf):
    """A filesystem or network call reachable from INPUT_TYPES. ComfyUI calls it
    on every /object_info, so a write there runs on every page load."""
    BAD = {"makedirs", "mkdir", "remove", "unlink", "rmtree", "write_text",
           "open", "touch", "rename", "replace"}
    out = []
    for name, cls in sorted(classes.items()):
        fn = getattr(cls, "INPUT_TYPES", None)
        if fn is None:
            continue
        # follow one level of helper calls defined in the same module
        try:
            src = textwrap.dedent(inspect.getsource(fn))
            mod = sys.modules.get(cls.__module__)
        except (OSError, TypeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = []
        for c in calls:
            f = c.func
            names.append(f.attr if isinstance(f, ast.Attribute) else
                         getattr(f, "id", None))
        for nm in names:
            if nm in BAD:
                out.append((name, f"INPUT_TYPES calls `{nm}` directly"))
        for nm in names:                     # one level down, same module
            helper = getattr(mod, nm, None) if nm else None
            if helper is None or not callable(helper):
                continue
            try:
                hsrc = textwrap.dedent(inspect.getsource(helper))
            except (OSError, TypeError):
                continue
            try:
                htree = ast.parse(hsrc)
            except SyntaxError:
                continue
            for n2 in ast.walk(htree):
                if isinstance(n2, ast.Call):
                    f2 = n2.func
                    a = f2.attr if isinstance(f2, ast.Attribute) else \
                        getattr(f2, "id", None)
                    if a in BAD and a != "open":
                        out.append((name, f"INPUT_TYPES -> {nm}() calls `{a}`"))
    return out


CHECKS = {"inputs": check_inputs, "gated": check_gated, "outputs": check_outputs,
          "labels": check_labels, "docs": check_docs, "bare": check_bare,
          "effects": check_effects}
TITLES = {
    "inputs":  "DEAD OR UNDECLARED INPUTS, AND OUTPUT ARITY",
    "gated":   "INPUTS READ ONLY UNDER ANOTHER INPUT'S SETTING (weak check — "
               "see the docstring)",
    "outputs": "OUTPUTS WIRED IN NO SHIPPED WORKFLOW",
    "labels":  "SAVED OUTPUT LABEL != DECLARED RETURN_NAME",
    "docs":    "DOCS NAMING SOMETHING THAT DOES NOT EXIST",
    "bare":    "INPUTS WITH NO TOOLTIP (advisory)",
    "effects": "SIDE EFFECTS IN INPUT_TYPES",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", default=",".join(CHECKS),
                    help="comma separated: " + ", ".join(CHECKS))
    ap.add_argument("--strict", default="inputs,docs,effects",
                    help="checks whose findings make the exit code non-zero")
    ap.add_argument("--quiet", action="store_true",
                    help="only print checks that found something")
    args = ap.parse_args()

    classes, failures = load_pack()
    wf = {}
    for f in sorted(glob.glob(str(ROOT / "workflows/**/*.json"), recursive=True)):
        try:
            wf[f] = json.loads(pathlib.Path(f).read_text())
        except Exception as e:
            print(f"unreadable workflow {pathlib.Path(f).name}: {e}")
    print(f"{len(classes)} node classes, {len(wf)} workflows")
    for n, e in failures:
        print(f"  MODULE FAILED TO IMPORT  {n}: {e[:80]}")
    if failures:
        print("  (a module that will not import is audited as zero nodes — fix "
              "the stub in load_pack() or the import)")

    wanted = [c.strip() for c in args.checks.split(",") if c.strip()]
    strict = {c.strip() for c in args.strict.split(",") if c.strip()}
    bad = [c for c in wanted if c not in CHECKS]
    if bad:
        sys.exit(f"unknown check(s): {', '.join(bad)}")
    rc, total = 0, 0
    for c in wanted:
        found = CHECKS[c](classes, wf)
        total += len(found)
        if args.quiet and not found:
            continue
        print()
        print("=" * 76)
        print(f"{TITLES[c]}  ({len(found)})")
        print("=" * 76)
        for node, detail in found:
            print(f"  {node:<30} {detail}")
        if not found:
            print("  clean")
        if found and c in strict:
            rc = 1
    print()
    print(f"{total} finding(s) across {len(wanted)} check(s)"
          + ("" if rc == 0 else "  — strict check(s) failed"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
