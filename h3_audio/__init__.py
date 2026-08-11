"""MiniMax-H3 audio prompt nodes for ComfyUI."""

import json
import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, PRESETS
from .video_nodes import (NODE_CLASS_MAPPINGS as _VID_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _VID_NAMES)
from .lint import (NODE_CLASS_MAPPINGS as _LINT_CLASSES,
                   NODE_DISPLAY_NAME_MAPPINGS as _LINT_NAMES)
from .characters import (NODE_CLASS_MAPPINGS as _CHAR_CLASSES,
                         NODE_DISPLAY_NAME_MAPPINGS as _CHAR_NAMES)
from .scene import (NODE_CLASS_MAPPINGS as _SCENE_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _SCENE_NAMES)
from .links import (NODE_CLASS_MAPPINGS as _LINK_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _LINK_NAMES)
from .tools import (NODE_CLASS_MAPPINGS as _TOOL_CLASSES,
                    NODE_DISPLAY_NAME_MAPPINGS as _TOOL_NAMES)
from .rewriter import (NODE_CLASS_MAPPINGS as _RW_CLASSES,
                       NODE_DISPLAY_NAME_MAPPINGS as _RW_NAMES)

# long-form video nodes share the pack; audio and video keep separate categories
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_VID_CLASSES, **_LINT_CLASSES,
                       **_CHAR_CLASSES, **_SCENE_CLASSES, **_LINK_CLASSES,
                       **_TOOL_CLASSES, **_RW_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_VID_NAMES, **_LINT_NAMES,
                              **_CHAR_NAMES, **_SCENE_NAMES, **_LINK_NAMES,
                              **_TOOL_NAMES, **_RW_NAMES}

# serves the widget-hiding / preset / plan-display extension
WEB_DIRECTORY = "./web"


def _register_routes():
    """Let the browser fetch a preset so it can fill the widgets visibly.

    The node also applies presets server-side, so the graph still renders correctly
    if this route is unavailable — this only exists so the fields VISIBLY populate
    and stay editable rather than being silently overridden at run time.
    """
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return

    routes = getattr(PromptServer.instance, "routes", None)
    if routes is None:
        return

    @routes.get("/h3_audio/preset")
    async def _preset(request):
        name = request.rel_url.query.get("name", "")
        data = PRESETS.get(name)
        if not data:
            return web.json_response({}, status=404)
        return web.json_response(data)


_register_routes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
