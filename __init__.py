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
from .budget import (NODE_CLASS_MAPPINGS as _BUDGET_CLASSES,
                     NODE_DISPLAY_NAME_MAPPINGS as _BUDGET_NAMES)
from .character import (NODE_CLASS_MAPPINGS as _CHAR_CLASSES,
                        NODE_DISPLAY_NAME_MAPPINGS as _CHAR_NAMES)
from .crop import (NODE_CLASS_MAPPINGS as _CROP_CLASSES,
                   NODE_DISPLAY_NAME_MAPPINGS as _CROP_NAMES)
from .longform import (NODE_CLASS_MAPPINGS as _LF_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _LF_NAMES)
from .mask import (NODE_CLASS_MAPPINGS as _MASK_CLASSES,
                   NODE_DISPLAY_NAME_MAPPINGS as _MASK_NAMES)
from .prompt_lint import (NODE_CLASS_MAPPINGS as _LINT_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _LINT_NAMES)
from .prompt_links import (NODE_CLASS_MAPPINGS as _LINK_CLASSES,
                           NODE_DISPLAY_NAME_MAPPINGS as _LINK_NAMES)
from .prompt_rewriter import (NODE_CLASS_MAPPINGS as _RW_CLASSES,
                              NODE_DISPLAY_NAME_MAPPINGS as _RW_NAMES)
from .prompt_scene import (NODE_CLASS_MAPPINGS as _SCENE_CLASSES,
                           NODE_DISPLAY_NAME_MAPPINGS as _SCENE_NAMES)
from .video import (NODE_CLASS_MAPPINGS as _VID_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _VID_NAMES)

_PARTS = (
    (_AUDIO_CLASSES, _AUDIO_NAMES),
    (_BUDGET_CLASSES, _BUDGET_NAMES),
    (_CHAR_CLASSES, _CHAR_NAMES),
    (_CROP_CLASSES, _CROP_NAMES),
    (_LF_CLASSES, _LF_NAMES),
    (_MASK_CLASSES, _MASK_NAMES),
    (_LINT_CLASSES, _LINT_NAMES),
    (_LINK_CLASSES, _LINK_NAMES),
    (_RW_CLASSES, _RW_NAMES),
    (_SCENE_CLASSES, _SCENE_NAMES),
    (_VID_CLASSES, _VID_NAMES),
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


_register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
