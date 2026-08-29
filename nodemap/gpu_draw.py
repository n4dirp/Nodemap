"""GPU drawing helpers for nodes minimap overlay."""

import math
from typing import Any

import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .theme import _srgb_to_linear, _theme_rgba

GPUStageInterfaceInfo = gpu.types.GPUStageInterfaceInfo
GPUShaderCreateInfo = gpu.types.GPUShaderCreateInfo

_FILL_SDF_SHADER: gpu.types.GPUShader | None = None
_FILL_SDF_VARYING_SHADER: gpu.types.GPUShader | None = None
_FILL_SDF_HOLE_SHADER: gpu.types.GPUShader | None = None
_FILL_SDF_CLIP_SHADER: gpu.types.GPUShader | None = None
_BORDER_SDF_SHADER: gpu.types.GPUShader | None = None
_PILL_SHADER: gpu.types.GPUShader | None = None
_PILL_BORDER_SHADER: gpu.types.GPUShader | None = None
_BATCH_PILL_SHADER: gpu.types.GPUShader | None = None
_BATCH_RECT_SHADER: gpu.types.GPUShader | None = None
_BATCH_RECT_BORDER_SHADER: gpu.types.GPUShader | None = None
_BATCH_NOODLE_SHADER: gpu.types.GPUShader | None = None

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

_FILL_FRAG_VARYING_SRC = """
float sdRoundRectVarying(vec2 p, vec2 b, vec4 r) {
    float radius = mix(
        mix(r.w, r.z, step(0.0, p.x)),
        mix(r.x, r.y, step(0.0, p.x)),
        step(0.0, p.y)
    );
    vec2 q = abs(p) - b + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}
void main() {
    float dist = sdRoundRectVarying(vUv, halfSize, radii);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(color.rgb, color.a * alpha);
}
"""

_FILL_FRAG_HOLE_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float outerDist = sdRoundRect(vUv, outerData.xy, outerData.z);
    float innerDist = sdRoundRect(vUv - innerOffset, innerHalfSize, outerData.w);
    float dist = max(outerDist, -innerDist);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(color.rgb, color.a * alpha);
}
"""

_FILL_FRAG_CLIP_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float fillDist = sdRoundRect(vUv, sizeData.xy, sizeData.z);
    float clipDist = sdRoundRect(vUv - clipData.xy, clipData.zw, sizeData.w);
    float dist = max(fillDist, clipDist);
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

_BATCH_PILL_VERT_SRC = """
void main() {
    vUv = uv;
    vHalfSize = halfSize;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_BATCH_PILL_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float r = min(vHalfSize.x, vHalfSize.y);
    float dist = sdRoundRect(vUv, vHalfSize, r);
    float alpha = 1.0 - smoothstep(-0.5, 0.5, dist);
    fragColor = vec4(color.rgb, color.a * alpha);
}
"""

_BATCH_RECT_VERT_SRC = """
void main() {
    vUv = uv;
    vHalfSize = halfSize;
    vRadius = radius;
    vColor = color;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_BATCH_RECT_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float dist = sdRoundRect(vUv, vHalfSize, vRadius);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(vColor.rgb, vColor.a * alpha);
}
"""

_BATCH_RECT_BORDER_VERT_SRC = """
void main() {
    vUv = uv;
    vHalfSize = halfSize;
    vRadius = radius;
    vColor = color;
    vLineWidth = lineWidth;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_BATCH_RECT_BORDER_FRAG_SRC = """
float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}
void main() {
    float bw = min(vLineWidth, min(vHalfSize.x, vHalfSize.y));
    float r2 = max(0.0, vRadius - bw);
    float outer = sdRoundRect(vUv, vHalfSize, vRadius);
    float inner = sdRoundRect(vUv, vHalfSize - bw, r2);
    float dist = max(outer, -inner);
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    fragColor = vec4(vColor.rgb, vColor.a * alpha);
}
"""

_BATCH_NOODLE_VERT_SRC = """
void main() {
    vT = uv.x;
    vPos = pos.xy;
    vSegA = segA;
    vSegB = segB;
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
}
"""

_BATCH_NOODLE_FRAG_SRC = """
void main() {
    // Per-quad capsule: the fragment measures distance to the quad's own chord
    // (segA->segB), interpolated by the along-chord fraction vT. Each rect is a
    // true spatial capsule around a chord that sits on the curve, so no curve
    // parameter matching is needed and the tube stays continuous across
    // segments. Adaptive subdivision keeps every chord within the AA band.
    float u = clamp(vT, 0.0, 1.0);
    vec2 q = mix(vSegA, vSegB, u);
    float d = length(vPos - q);
    float sd = d - halfThick;
    float alpha = 1.0 - smoothstep(-0.5, 0.5, sd);
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


def _get_sdf_fill_varying_shader() -> gpu.types.GPUShader:
    global _FILL_SDF_VARYING_SHADER
    if _FILL_SDF_VARYING_SHADER is None:
        vert_out = GPUStageInterfaceInfo("fill_varying_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC2", "halfSize")
        info.push_constant("VEC4", "radii")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_FILL_FRAG_VARYING_SRC)
        _FILL_SDF_VARYING_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _FILL_SDF_VARYING_SHADER


def _get_sdf_fill_hole_shader() -> gpu.types.GPUShader:
    global _FILL_SDF_HOLE_SHADER
    if _FILL_SDF_HOLE_SHADER is None:
        vert_out = GPUStageInterfaceInfo("fill_hole_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC4", "outerData")
        info.push_constant("VEC2", "innerOffset")
        info.push_constant("VEC2", "innerHalfSize")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_FILL_FRAG_HOLE_SRC)
        _FILL_SDF_HOLE_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _FILL_SDF_HOLE_SHADER


def _get_sdf_fill_clip_shader() -> gpu.types.GPUShader:
    global _FILL_SDF_CLIP_SHADER
    if _FILL_SDF_CLIP_SHADER is None:
        vert_out = GPUStageInterfaceInfo("fill_clip_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC4", "sizeData")
        info.push_constant("VEC4", "clipData")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_FILL_FRAG_CLIP_SRC)
        _FILL_SDF_CLIP_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _FILL_SDF_CLIP_SHADER


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


def _get_batch_pill_shader() -> gpu.types.GPUShader:
    """Pill SDF shader taking *halfSize* as a per-vertex attribute (for batching)."""
    global _BATCH_PILL_SHADER
    if _BATCH_PILL_SHADER is None:
        vert_out = GPUStageInterfaceInfo("batch_pill_iface")
        vert_out.smooth("VEC2", "vUv")
        vert_out.smooth("VEC2", "vHalfSize")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_in(2, "VEC2", "halfSize")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_BATCH_PILL_VERT_SRC)
        info.fragment_source(_BATCH_PILL_FRAG_SRC)
        _BATCH_PILL_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _BATCH_PILL_SHADER


def _get_batch_rect_shader() -> gpu.types.GPUShader:
    """Custom batched rounded rectangle background fill shader."""
    global _BATCH_RECT_SHADER
    if _BATCH_RECT_SHADER is None:
        vert_out = GPUStageInterfaceInfo("batch_rect_iface")
        vert_out.smooth("VEC2", "vUv")
        vert_out.smooth("VEC2", "vHalfSize")
        vert_out.smooth("FLOAT", "vRadius")
        vert_out.smooth("VEC4", "vColor")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_in(2, "VEC2", "halfSize")
        info.vertex_in(3, "FLOAT", "radius")
        info.vertex_in(4, "VEC4", "color")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_BATCH_RECT_VERT_SRC)
        info.fragment_source(_BATCH_RECT_FRAG_SRC)
        _BATCH_RECT_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _BATCH_RECT_SHADER


def _get_batch_rect_border_shader() -> gpu.types.GPUShader:
    """Custom batched rounded rectangle border shader."""
    global _BATCH_RECT_BORDER_SHADER
    if _BATCH_RECT_BORDER_SHADER is None:
        vert_out = GPUStageInterfaceInfo("batch_rect_border_iface")
        vert_out.smooth("VEC2", "vUv")
        vert_out.smooth("VEC2", "vHalfSize")
        vert_out.smooth("FLOAT", "vRadius")
        vert_out.smooth("VEC4", "vColor")
        vert_out.smooth("FLOAT", "vLineWidth")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_in(2, "VEC2", "halfSize")
        info.vertex_in(3, "FLOAT", "radius")
        info.vertex_in(4, "VEC4", "color")
        info.vertex_in(5, "FLOAT", "lineWidth")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_BATCH_RECT_BORDER_VERT_SRC)
        info.fragment_source(_BATCH_RECT_BORDER_FRAG_SRC)
        _BATCH_RECT_BORDER_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _BATCH_RECT_BORDER_SHADER


def _get_batch_noodle_shader() -> gpu.types.GPUShader:
    """Batched curved noodle shader measuring distance to each quad's chord."""

    global _BATCH_NOODLE_SHADER
    if _BATCH_NOODLE_SHADER is None:
        vert_out = GPUStageInterfaceInfo("batch_noodle_iface")
        vert_out.smooth("FLOAT", "vT")
        vert_out.smooth("VEC2", "vPos")
        vert_out.smooth("VEC2", "vSegA")
        vert_out.smooth("VEC2", "vSegB")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("FLOAT", "halfThick")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_in(2, "VEC2", "segA")
        info.vertex_in(3, "VEC2", "segB")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_BATCH_NOODLE_VERT_SRC)
        info.fragment_source(_BATCH_NOODLE_FRAG_SRC)
        _BATCH_NOODLE_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _BATCH_NOODLE_SHADER


def _build_pill_batch(
    wires: list[tuple[float, float, float, float]],
    thickness: float,
) -> tuple[Any, Any]:
    """Bake pill-shaped wires into a GPU batch; returns ``(shader, batch)``.

    Both are None when *wires* is empty. The color is left as a draw-time
    uniform so one batch per distinct color suffices.
    """
    if not wires:
        return None, None

    shader = _get_batch_pill_shader()
    pad = 2.0
    hh = thickness / 2

    all_pos: list[tuple[float, float, float]] = []
    all_uv: list[tuple[float, float]] = []
    all_half_size: list[tuple[float, float]] = []

    for mx, my, length, angle in wires:
        hw = length / 2
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Local-space corners with padding for AA margin
        lx0 = -hw - pad
        ly0 = -hh - pad
        lx1 = hw + pad
        ly1 = hh + pad

        # Pre-transform corners to world space (rotate + translate)
        all_pos.extend(
            [
                (mx + lx0 * cos_a - ly0 * sin_a, my + lx0 * sin_a + ly0 * cos_a, 0.0),
                (mx + lx1 * cos_a - ly0 * sin_a, my + lx1 * sin_a + ly0 * cos_a, 0.0),
                (mx + lx1 * cos_a - ly1 * sin_a, my + lx1 * sin_a + ly1 * cos_a, 0.0),
                (mx + lx0 * cos_a - ly1 * sin_a, my + lx0 * sin_a + ly1 * cos_a, 0.0),
            ]
        )

        all_uv.extend(
            [
                (lx0, ly0),
                (lx1, ly0),
                (lx1, ly1),
                (lx0, ly1),
            ]
        )

        all_half_size.extend([(hw, hh)] * 4)

    # Build index buffer: 2 triangles per quad, 4 verts per wire
    indices: list[tuple[int, int, int]] = []
    for i in range(len(wires)):
        base = i * 4
        indices.append((base, base + 1, base + 2))
        indices.append((base + 2, base + 3, base))

    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": all_pos, "uv": all_uv, "halfSize": all_half_size},
        indices=indices,
    )
    return shader, batch


def _build_noodle_batch(
    wires: list[tuple[float, float, float, float, float, float, float, float]],
    half_thick: float,
) -> tuple[Any, Any]:
    """Bake curved noodle wires as triangle strips; returns ``(shader, batch)``.

    Each *wire* is ``(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y)`` in baked space
    defining a cubic Bezier with horizontal handles (Blender's ease-in /
    ease-out style). The wire is subdivided so every chord stays within the AA
    band of the cubic; each chord emits a rectangle. The fragment resolves each
    rectangle as a capsule SDF against its own chord (``segA``/``segB`` shared
    by all four corners), with round end caps emitted as endpoint squares. This
    measures true spatial distance, so the tube is continuous and follows the
    curve without parametric mismatch. Color stays as a draw-time uniform.
    """

    if not wires:
        return None, None

    shader = _get_batch_noodle_shader()
    pad = 2.0
    lateral = half_thick + pad

    all_pos: list[tuple[float, float, float]] = []
    all_uv: list[tuple[float, float]] = []
    all_segA: list[tuple[float, float]] = []
    all_segB: list[tuple[float, float]] = []

    indices: list[tuple[int, int, int]] = []
    base_vert = 0

    for p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y in wires:
        # Estimate length for tessellation density.
        d01 = math.hypot(p1x - p0x, p1y - p0y)
        d12 = math.hypot(p2x - p1x, p2y - p1y)
        d23 = math.hypot(p3x - p2x, p3y - p2y)
        est = d01 + d12 + d23
        # Per-chord capsule SDF is spatially exact, but a curved segment's tube
        # bulges beyond its chord by the sagitta. Subdivide while that sagitta
        # exceeds a tolerance so the tube follows the curve; the coarser 1.5
        # tolerance keeps the chord count low at the cost of a sub-pixel thinning
        # on the inside of only the tightest turns. Caps bound worst-case cost.
        n = int(est / 10.0) + 2
        n = max(2, min(n, 16))

        def _bezier_at(t):
            om = 1.0 - t
            om2 = om * om
            t2 = t * t
            return (
                om2 * om * p0x + 3.0 * om2 * t * p1x + 3.0 * om * t2 * p2x + t2 * t * p3x,
                om2 * om * p0y + 3.0 * om2 * t * p1y + 3.0 * om * t2 * p2y + t2 * t * p3y,
            )

        step = 1.0 / n
        t_list = [i * step for i in range(n + 1)]
        i = 0
        while i < len(t_list) - 1 and len(t_list) < 32:
            t0 = t_list[i]
            t1 = t_list[i + 1]
            if t1 - t0 <= 1.0 / 32.0:
                i += 1
                continue
            ca = _bezier_at(t0)
            cb = _bezier_at(t1)
            cm = _bezier_at((t0 + t1) * 0.5)
            dev = math.hypot(cm[0] - (ca[0] + cb[0]) * 0.5, cm[1] - (ca[1] + cb[1]) * 0.5)
            if dev > 1.5:
                t_list.insert(i + 1, (t0 + t1) * 0.5)
            else:
                i += 1

        centers = [_bezier_at(t) for t in t_list]
        n = len(t_list) - 1

        def _unit(a, b):
            dx_ = b[0] - a[0]
            dy_ = b[1] - a[1]
            dl_ = math.hypot(dx_, dy_)
            if dl_ > 1e-9:
                return (dx_ / dl_, dy_ / dl_)
            return (1.0, 0.0)

        dchord = _unit(centers[0], centers[1])
        dchord_n = (-dchord[1], dchord[0])

        # Emit one rectangle for a chord. All four vertices share the same
        # segA/segB anchors (the chord endpoints) so the fragment measures a
        # true capsule around exactly that chord; uv.x interpolates the
        # along-chord fraction u.
        def _emit_rect(ax_, ay_, bx_, by_, nx, ny):
            nonlocal base_vert
            lx0 = ax_ + nx * lateral
            ly0 = ay_ + ny * lateral
            rx0 = ax_ - nx * lateral
            ry0 = ay_ - ny * lateral
            lx1 = bx_ + nx * lateral
            ly1 = by_ + ny * lateral
            rx1 = bx_ - nx * lateral
            ry1 = by_ - ny * lateral
            base = base_vert
            all_pos.extend([(lx0, ly0, 0.0), (rx0, ry0, 0.0), (lx1, ly1, 0.0), (rx1, ry1, 0.0)])
            all_uv.extend([(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
            all_segA.extend([(ax_, ay_)] * 4)
            all_segB.extend([(bx_, by_)] * 4)
            indices.append((base, base + 1, base + 2))
            indices.append((base + 2, base + 1, base + 3))
            base_vert += 4

        # End-cap squares: a full disk around each endpoint collapses both segA
        # and segB to the endpoint, so every fragment inside resolves distance to
        # that point (a round cap) regardless of direction. Robust for tiny/steep
        # wires where lateral dwarfs the chord.
        def _emit_square(cx, cy):
            nonlocal base_vert
            base = base_vert
            corners = [
                (cx + dchord[0] * lateral, cy + dchord[1] * lateral),
                (cx + dchord_n[0] * lateral, cy + dchord_n[1] * lateral),
                (cx - dchord[0] * lateral, cy - dchord[1] * lateral),
                (cx - dchord_n[0] * lateral, cy - dchord_n[1] * lateral),
            ]
            for c in corners:
                all_pos.append((c[0], c[1], 0.0))
                all_uv.append((0.0, 0.0))
                all_segA.append((cx, cy))
                all_segB.append((cx, cy))
            indices.append((base, base + 1, base + 2))
            indices.append((base + 2, base + 3, base))
            base_vert += 4

        _emit_square(centers[0][0], centers[0][1])

        # Interior quads: one rectangle per chord, aligned to that chord's own
        # normal so the capsule follows the curve (the cap squares already cover
        # the first/last endpoint disks). Each chord extends past its shared
        # joints by `extend` so adjacent capsules overlap their straight sides,
        # filling the small rounded wedge that shows on the outside of a turn.
        # The wire's first and last ends stay at the true endpoints.
        extend = 0.175
        prev_nx, prev_ny = 0.0, 1.0
        for qi in range(len(centers) - 1):
            ax_, ay_ = centers[qi]
            bx_, by_ = centers[qi + 1]
            vx = bx_ - ax_
            vy = by_ - ay_
            cl = math.hypot(vx, vy)
            if cl > 1e-9:
                nx = -vy / cl
                ny = vx / cl
                prev_nx, prev_ny = nx, ny
                ux, uy = vx / cl, vy / cl
            else:
                nx, ny = prev_nx, prev_ny
                ux, uy = 0.0, 0.0
            if qi > 0:
                ax_ -= ux * extend
                ay_ -= uy * extend
            if qi < len(centers) - 2:
                bx_ += ux * extend
                by_ += uy * extend
            _emit_rect(ax_, ay_, bx_, by_, nx, ny)

        _emit_square(centers[n][0], centers[n][1])

    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": all_pos, "uv": all_uv, "segA": all_segA, "segB": all_segB},
        indices=indices,
    )
    return shader, batch


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


def _draw_text_with_shadow(
    font_id: int, text: str, x: float, y: float, color: tuple[float, ...], size: int, with_shadow: bool = True
):
    if len(color) == 3:
        color = color + (1.0,)
    blf.size(font_id, size)
    if with_shadow:
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0, 0, 0, 255)
        blf.shadow_offset(font_id, 0, -1)
    blf.position(font_id, x, y, 0)
    blf.color(font_id, *color)
    blf.draw(font_id, text)
    if with_shadow:
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


def _draw_filled_rounded_rect_varying(x, y, w, h, radii, color):
    if w <= 0 or h <= 0:
        return
    max_r = min(w / 2, h / 2)
    radii = tuple(max(0, min(r, max_r)) for r in radii)

    shader = _get_sdf_fill_varying_shader()
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
    shader.uniform_float("radii", radii)
    batch.draw(shader)


def _draw_filled_rounded_rect_with_hole(
    mx,
    my,
    mw,
    mh,
    outer_r,
    ix,
    iy,
    iw,
    ih,
    inner_r,
    color,
):
    if mw <= 0 or mh <= 0 or iw <= 0 or ih <= 0:
        return
    outer_r = max(0, min(outer_r, mw / 2, mh / 2))

    shader = _get_sdf_fill_hole_shader()
    hw, hh = mw / 2, mh / 2

    inner_off_x = (ix + iw / 2) - (mx + mw / 2)
    inner_off_y = (iy + ih / 2) - (my + mh / 2)
    inner_hw = iw / 2
    inner_hh = ih / 2

    vertices = (
        (mx, my, 0.0),
        (mx + mw, my, 0.0),
        (mx + mw, my + mh, 0.0),
        (mx, my + mh, 0.0),
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
    shader.uniform_float("outerData", (hw, hh, outer_r, inner_r))
    shader.uniform_float("innerOffset", (inner_off_x, inner_off_y))
    shader.uniform_float("innerHalfSize", (inner_hw, inner_hh))
    batch.draw(shader)


def _draw_filled_rounded_rect_clipped(x, y, w, h, r, color, clip_x, clip_y, clip_w, clip_h, clip_r):
    """Draw a rounded rect fill intersected with a rounded clip region."""
    if w <= 0 or h <= 0 or clip_w <= 0 or clip_h <= 0:
        return
    r = max(0, min(r, w / 2, h / 2))
    clip_r = max(0, min(clip_r, clip_w / 2, clip_h / 2))

    shader = _get_sdf_fill_clip_shader()
    hw, hh = w / 2, h / 2
    cx, cy = x + hw, y + hh
    clip_hw, clip_hh = clip_w / 2, clip_h / 2
    off_x = (clip_x + clip_hw) - cx
    off_y = (clip_y + clip_hh) - cy

    # Pad the quad so the clip arc's AA falloff has room at shared edges
    pad = 2.0
    px0, py0 = x - pad, y - pad
    px1, py1 = x + w + pad, y + h + pad

    vertices = (
        (px0, py0, 0.0),
        (px1, py0, 0.0),
        (px1, py1, 0.0),
        (px0, py1, 0.0),
    )
    uvs = (
        (px0 - cx, py0 - cy),
        (px1 - cx, py0 - cy),
        (px1 - cx, py1 - cy),
        (px0 - cx, py1 - cy),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("sizeData", (hw, hh, r, clip_r))
    shader.uniform_float("clipData", (off_x, off_y, clip_hw, clip_hh))
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
