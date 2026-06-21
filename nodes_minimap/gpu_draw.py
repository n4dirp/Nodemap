"""GPU drawing helpers for nodes minimap overlay."""

import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .helpers import _srgb_to_linear, _theme

GPUStageInterfaceInfo = gpu.types.GPUStageInterfaceInfo
GPUShaderCreateInfo = gpu.types.GPUShaderCreateInfo

_FILL_SDF_SHADER: gpu.types.GPUShader | None = None
_BORDER_SDF_SHADER: gpu.types.GPUShader | None = None
_PILL_SHADER: gpu.types.GPUShader | None = None
_PILL_BORDER_SHADER: gpu.types.GPUShader | None = None

_FILL_VERT_SRC = """
void main() {
    vUv = uv;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_FILL_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float dist = sdRoundRect(vUv, halfSize, radius);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(color.rgb, color.a * alpha);
}
"""

_BORDER_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float bw = min(lineWidth, min(halfSize.x, halfSize.y));
    float r2 = max(0.0, radius - bw);
    float outer = sdRoundRect(vUv, halfSize, radius);
    float inner = sdRoundRect(vUv, halfSize - bw, r2);
    float dist = max(outer, -inner);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(color.rgb, color.a * alpha);
}
"""

_PILL_FILL_VERT_SRC = """
void main() {
    vUv = uv;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_PILL_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float r = min(halfSize.x, halfSize.y);
    float dist = sdRoundRect(vUv, halfSize, r);

    float alpha = 1.0 - smoothstep(-0.5, 0.5, dist);

    fragColor = vec4(color.rgb, color.a * alpha);
}
"""

_PILL_BORDER_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float r = min(halfSize.x, halfSize.y);
    float bw = min(lineWidth, min(halfSize.x, halfSize.y));
    float r2 = max(0.0, r - bw);
    float outer = sdRoundRect(vUv, halfSize, r);
    float inner = sdRoundRect(vUv, halfSize - bw, r2);
    float dist = max(outer, -inner);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(color.rgb, color.a * alpha);
}
"""


def _get_sdf_fill_shader() -> gpu.types.GPUShader:
    global _FILL_SDF_SHADER
    if _FILL_SDF_SHADER is None:
        vert_out = GPUStageInterfaceInfo("fill_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC2", "halfSize")
        info.push_constant("FLOAT", "radius")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_FILL_FRAG_SRC)
        _FILL_SDF_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _FILL_SDF_SHADER


def _get_sdf_border_shader() -> gpu.types.GPUShader:
    global _BORDER_SDF_SHADER
    if _BORDER_SDF_SHADER is None:
        vert_out = GPUStageInterfaceInfo("border_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC2", "halfSize")
        info.push_constant("FLOAT", "radius")
        info.push_constant("FLOAT", "lineWidth")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_BORDER_FRAG_SRC)
        _BORDER_SDF_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _BORDER_SDF_SHADER


def _get_pill_shader() -> gpu.types.GPUShader:
    global _PILL_SHADER
    if _PILL_SHADER is None:
        vert_out = GPUStageInterfaceInfo("pill_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC2", "halfSize")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_PILL_FILL_VERT_SRC)
        info.fragment_source(_PILL_FRAG_SRC)
        _PILL_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _PILL_SHADER


def _get_pill_border_shader() -> gpu.types.GPUShader:
    global _PILL_BORDER_SHADER
    if _PILL_BORDER_SHADER is None:
        vert_out = GPUStageInterfaceInfo("pill_border_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC2", "halfSize")
        info.push_constant("FLOAT", "lineWidth")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_PILL_FILL_VERT_SRC)
        info.fragment_source(_PILL_BORDER_FRAG_SRC)
        _PILL_BORDER_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _PILL_BORDER_SHADER


def _theme_rgba(path: str, default: tuple[float, ...]):
    """Fetch a theme color via _theme, guaranteeing a 4-element RGBA tuple."""
    result = _theme(path, default)
    if len(result) == 3:
        return result + (1.0,)
    return result


def _get_theme_colors():
    return {
        "bg_color": _theme_rgba("user_interface.wcol_toolbar_item.inner", (0.25, 0.25, 0.25, 1.0)),
        "panel_border": _theme_rgba("user_interface.wcol_toolbar_item.outline", (1.0, 1.0, 1.0, 0.02)),
        "tile_default": _theme_rgba("user_interface.wcol_regular.inner", (0.25, 0.25, 0.25, 1.0)),
        "tile_picked": _theme_rgba("user_interface.wcol_regular.inner_sel", (0.28, 0.45, 0.7, 1.0)),
        "border_active": _theme_rgba("view_3d.object_active", (1.0, 0.63, 0.16, 1.0)),
        "tile_border": _theme_rgba("user_interface.wcol_regular.outline", (1.0, 1.0, 1.0, 0.02)),
        "scroll_bar": _theme_rgba("user_interface.wcol_scroll.item", (0.35, 0.35, 0.35, 0.75)),
        "text": _theme_rgba("user_interface.wcol_regular.text_sel", (1.0, 1.0, 1.0, 1.0)),
        "info_text": _theme_rgba("view_3d.space.text_hi", (1.0, 1.0, 1.0, 1.0)),
        "tile_text": _theme_rgba("user_interface.wcol_regular.text_sel", (1.0, 1.0, 1.0, 1.0)),
        "tile_text_inactive": _theme_rgba("user_interface.wcol_regular.text", (1.0, 1.0, 1.0, 1.0)),
    }


def _draw_text_with_shadow(font_id: int, text: str, x: float, y: float, color: tuple[float, ...], size: int):
    if len(color) == 3:
        color = color + (1.0,)
    blf.size(font_id, size)
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0, 0, 0, 255)
    blf.shadow_offset(font_id, 0, -1)
    blf.position(font_id, x, y, 0)
    blf.color(font_id, *color)
    blf.draw(font_id, text)
    blf.disable(font_id, blf.SHADOW)


def _draw_filled_rounded_rect(x, y, w, h, r, color):
    if w <= 0 or h <= 0:
        return
    r = max(0, min(r, w / 2, h / 2))

    shader = _get_sdf_fill_shader()
    hw, hh = w / 2, h / 2

    vertices = (
        (x, y, 0.0),
        (x + w, y, 0.0),
        (x + w, y + h, 0.0),
        (x, y + h, 0.0),
    )
    uvs = (
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (hw, hh))
    shader.uniform_float("radius", r)
    batch.draw(shader)


def _draw_rounded_rect_border(x, y, w, h, r, color, line_width=1.0):
    if w <= 0 or h <= 0:
        return
    r = max(0, min(r, w / 2, h / 2))

    shader = _get_sdf_border_shader()
    hw, hh = w / 2, h / 2

    vertices = (
        (x, y, 0.0),
        (x + w, y, 0.0),
        (x + w, y + h, 0.0),
        (x, y + h, 0.0),
    )
    uvs = (
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (hw, hh))
    shader.uniform_float("radius", r)
    shader.uniform_float("lineWidth", line_width)
    batch.draw(shader)


def _draw_pill(x, y, w, h, color):
    if w <= 0 or h <= 0:
        return

    shader = _get_pill_shader()
    hw, hh = w / 2, h / 2

    pad = 2.0

    vertices = (
        (x - pad, y - pad, 0.0),
        (x + w + pad, y - pad, 0.0),
        (x + w + pad, y + h + pad, 0.0),
        (x - pad, y + h + pad, 0.0),
    )
    uvs = (
        (-hw - pad, -hh - pad),
        (hw + pad, -hh - pad),
        (hw + pad, hh + pad),
        (-hw - pad, hh + pad),
    )

    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )

    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (hw, hh))

    batch.draw(shader)


def _draw_pill_border(x, y, w, h, color, line_width=1.0):
    if w <= 0 or h <= 0:
        return

    shader = _get_pill_border_shader()
    hw, hh = w / 2, h / 2

    pad = 2.0

    vertices = (
        (x - pad, y - pad, 0.0),
        (x + w + pad, y - pad, 0.0),
        (x + w + pad, y + h + pad, 0.0),
        (x - pad, y + h + pad, 0.0),
    )
    uvs = (
        (-hw - pad, -hh - pad),
        (hw + pad, -hh - pad),
        (hw + pad, hh + pad),
        (-hw - pad, hh + pad),
    )

    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (hw, hh))
    shader.uniform_float("lineWidth", line_width)

    batch.draw(shader)
