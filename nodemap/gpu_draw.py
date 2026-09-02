"""Provide GPU drawing helpers for the minimap overlay."""

import math
from typing import Any

import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .theme import _srgb_to_linear

GPUStageInterfaceInfo = gpu.types.GPUStageInterfaceInfo
GPUShaderCreateInfo = gpu.types.GPUShaderCreateInfo

_FILL_SDF_SHADER: gpu.types.GPUShader | None = None
_FILL_SDF_VARYING_SHADER: gpu.types.GPUShader | None = None
_FILL_SDF_HOLE_SHADER: gpu.types.GPUShader | None = None
_FILL_SDF_CLIP_SHADER: gpu.types.GPUShader | None = None
_BORDER_SDF_SHADER: gpu.types.GPUShader | None = None
_BORDER_SDF_VARYING_SIDES_SHADER: gpu.types.GPUShader | None = None
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

_BORDER_FRAG_VARYING_SIDES_SRC = """
float sdRoundRectVarying(vec2 p, vec2 b, vec4 r) {
    float radius = mix(
        mix(r.w, r.z, step(0.0, p.x)),
        mix(r.x, r.y, step(0.0, p.x)),
        step(0.0, p.y)
    );
    vec2 q = abs(p) - b + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}
float sdRoundBoxAniso(vec2 p, vec2 bNeg, vec2 bPos, vec4 radii) {
    vec2 b = mix(bNeg, bPos, step(0.0, p));
    float radius = mix(
        mix(radii.w, radii.z, step(0.0, p.x)),
        mix(radii.x, radii.y, step(0.0, p.x)),
        step(0.0, p.y)
    );
    vec2 q = abs(p) - b + radius;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}
void main() {
    float lineWidth = lineData.x;
    float skipLeft = lineData.y;
    float skipRight = lineData.z;
    float bw = min(lineWidth, min(halfSize.x, halfSize.y));
    vec4 innerRadii = max(vec4(0.0), radii - bw);
    // Inner extents: shrink by bw on every side except the masked vertical
    // sides, where the band collapses so no stroke is emitted there. Two
    // adjacent buttons therefore contribute a single seam line instead of
    // two coincident strokes doubling its weight.
    vec2 bNeg = vec2(halfSize.x - bw * (1.0 - skipLeft), halfSize.y - bw);
    vec2 bPos = vec2(halfSize.x - bw * (1.0 - skipRight), halfSize.y - bw);
    float outer = sdRoundRectVarying(vUv, halfSize, radii);
    float inner = sdRoundBoxAniso(vUv, bNeg, bPos, innerRadii);
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


def _get_sdf_border_varying_sides_shader() -> gpu.types.GPUShader:
    """Return a border SDF shader that suppresses the vertical stroke on masked sides."""
    global _BORDER_SDF_VARYING_SIDES_SHADER
    if _BORDER_SDF_VARYING_SIDES_SHADER is None:
        vert_out = GPUStageInterfaceInfo("border_varying_sides_iface")
        vert_out.smooth("VEC2", "vUv")
        info = GPUShaderCreateInfo()
        info.push_constant("MAT4", "ModelViewProjectionMatrix")
        info.push_constant("VEC4", "color")
        info.push_constant("VEC2", "halfSize")
        info.push_constant("VEC4", "radii")
        info.push_constant("VEC4", "lineData")
        info.vertex_in(0, "VEC3", "pos")
        info.vertex_in(1, "VEC2", "uv")
        info.vertex_out(vert_out)
        info.fragment_out(0, "VEC4", "fragColor")
        info.vertex_source(_FILL_VERT_SRC)
        info.fragment_source(_BORDER_FRAG_VARYING_SIDES_SRC)
        _BORDER_SDF_VARYING_SIDES_SHADER = gpu.shader.create_from_info(info)
        del vert_out, info
    return _BORDER_SDF_VARYING_SIDES_SHADER


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
    """Return a pill SDF shader that takes *halfSize* as a per-vertex attribute."""
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
    """Return a batched rounded-rectangle background fill shader."""
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
    """Return a batched rounded-rectangle border shader."""
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
    """Return a batched curved noodle shader that measures distance to each quad's chord."""

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
    """Bake pill-shaped wires into a GPU batch and return ``(shader, batch)``.

    Return None for both when *wires* is empty. The color remains a draw-time
    uniform so one batch per distinct color suffices.
    """
    if not wires:
        return None, None

    shader = _get_batch_pill_shader()
    aa_pad = 2.0
    half_thickness = thickness / 2

    all_pos: list[tuple[float, float, float]] = []
    all_uv: list[tuple[float, float]] = []
    all_half_size: list[tuple[float, float]] = []
    indices: list[tuple[int, int, int]] = []
    base_vert = 0

    for seg_x, seg_y, length, angle in wires:
        half_length = length / 2
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        corner_x0 = -half_length - aa_pad
        corner_y0 = -half_thickness - aa_pad
        corner_x1 = half_length + aa_pad
        corner_y1 = half_thickness + aa_pad

        all_pos.extend(
            [
                (
                    seg_x + corner_x0 * cos_angle - corner_y0 * sin_angle,
                    seg_y + corner_x0 * sin_angle + corner_y0 * cos_angle,
                    0.0,
                ),
                (
                    seg_x + corner_x1 * cos_angle - corner_y0 * sin_angle,
                    seg_y + corner_x1 * sin_angle + corner_y0 * cos_angle,
                    0.0,
                ),
                (
                    seg_x + corner_x1 * cos_angle - corner_y1 * sin_angle,
                    seg_y + corner_x1 * sin_angle + corner_y1 * cos_angle,
                    0.0,
                ),
                (
                    seg_x + corner_x0 * cos_angle - corner_y1 * sin_angle,
                    seg_y + corner_x0 * sin_angle + corner_y1 * cos_angle,
                    0.0,
                ),
            ]
        )

        all_uv.extend(
            [
                (corner_x0, corner_y0),
                (corner_x1, corner_y0),
                (corner_x1, corner_y1),
                (corner_x0, corner_y1),
            ]
        )

        all_half_size.extend([(half_length, half_thickness)] * 4)

        base = base_vert
        indices.append((base, base + 1, base + 2))
        indices.append((base + 2, base + 3, base))
        base_vert += 4

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
    """Bake curved noodle wires as capsule quads and return ``(shader, batch)``.

    Each *wire* is ``(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y)`` in baked space
    defining a cubic Bezier with horizontal handles. The curve is uniformly
    subdivided into ``n`` chords (no adaptive refinement) and each chord emits
    a capsule rectangle whose fragment measures distance to that chord. Round
    end caps are emitted as endpoint squares.
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
    hypot = math.hypot

    for p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y in wires:
        seg_len_01 = hypot(p1x - p0x, p1y - p0y)
        seg_len_12 = hypot(p2x - p1x, p2y - p1y)
        seg_len_23 = hypot(p3x - p2x, p3y - p2y)
        est_length = seg_len_01 + seg_len_12 + seg_len_23
        # Uniform subdivision only — the per-quad capsule shader is exact for
        # its chord, so a modest fixed count keeps the tube visually continuous.
        # Using 14 px per segment and capping at 8 keeps the vertex count ~2x
        # lower than the previous adaptive 16+32 path while preserving shape at
        # minimap scale (sub-pixel sagitta error < 1 px for typical bakes).
        n = int(est_length / 14.0) + 2
        if n < 2:
            n = 2
        elif n > 10:
            n = 10
        inv_n = 1.0 / n

        # Sample the Bezier uniformly — inline the cubic to avoid closure
        # overhead and repeated function calls.
        centers: list[tuple[float, float]] = [None] * (n + 1)  # type: ignore[list-item]
        for s in range(n + 1):
            t = s * inv_n
            omt = 1.0 - t
            omt2 = omt * omt
            t2 = t * t
            omt3 = omt2 * omt
            t3 = t2 * t
            # Horner-like cubic Bezier
            x = omt3 * p0x + 3.0 * omt2 * t * p1x + 3.0 * omt * t2 * p2x + t3 * p3x
            y = omt3 * p0y + 3.0 * omt2 * t * p1y + 3.0 * omt * t2 * p2y + t3 * p3y
            centers[s] = (x, y)

        # First chord direction for end-cap orientation.
        dx0 = centers[1][0] - centers[0][0]
        dy0 = centers[1][1] - centers[0][1]
        d0 = hypot(dx0, dy0)
        if d0 > 1e-9:
            chord_dir_x, chord_dir_y = dx0 / d0, dy0 / d0
        else:
            chord_dir_x, chord_dir_y = 1.0, 0.0
        chord_dir_nx, chord_dir_ny = -chord_dir_y, chord_dir_x

        # End-cap squares — collapsed chord (segA == segB == cap).
        def _emit_cap(cap_x: float, cap_y: float) -> None:
            nonlocal base_vert
            base = base_vert
            all_pos.append((cap_x + chord_dir_x * lateral, cap_y + chord_dir_y * lateral, 0.0))
            all_pos.append((cap_x + chord_dir_nx * lateral, cap_y + chord_dir_ny * lateral, 0.0))
            all_pos.append((cap_x - chord_dir_x * lateral, cap_y - chord_dir_y * lateral, 0.0))
            all_pos.append((cap_x - chord_dir_nx * lateral, cap_y - chord_dir_ny * lateral, 0.0))
            all_uv.extend([(0.0, 0.0)] * 4)
            all_segA.extend([(cap_x, cap_y)] * 4)
            all_segB.extend([(cap_x, cap_y)] * 4)
            indices.append((base, base + 1, base + 2))
            indices.append((base + 2, base + 3, base))
            base_vert += 4

        _emit_cap(centers[0][0], centers[0][1])

        # Interior quads — one per chord, inlined to avoid per-chord closures.
        chord_extend = 0.01
        prev_nx, prev_ny = 0.0, 1.0
        for chord_index in range(n):
            ax_, ay_ = centers[chord_index]
            bx_, by_ = centers[chord_index + 1]
            dir_x = bx_ - ax_
            dir_y = by_ - ay_
            dir_len = hypot(dir_x, dir_y)
            if dir_len > 1e-9:
                nx = -dir_y / dir_len
                ny = dir_x / dir_len
                prev_nx, prev_ny = nx, ny
                ux, uy = dir_x / dir_len, dir_y / dir_len
            else:
                nx, ny = prev_nx, prev_ny
                ux, uy = 0.0, 0.0
            if chord_index > 0:
                ax_ -= ux * chord_extend
                ay_ -= uy * chord_extend
            if chord_index < n - 1:
                bx_ += ux * chord_extend
                by_ += uy * chord_extend
            left_x0 = ax_ + nx * lateral
            left_y0 = ay_ + ny * lateral
            right_x0 = ax_ - nx * lateral
            right_y0 = ay_ - ny * lateral
            left_x1 = bx_ + nx * lateral
            left_y1 = by_ + ny * lateral
            right_x1 = bx_ - nx * lateral
            right_y1 = by_ - ny * lateral
            base = base_vert
            all_pos.extend(
                [(left_x0, left_y0, 0.0), (right_x0, right_y0, 0.0), (left_x1, left_y1, 0.0), (right_x1, right_y1, 0.0)]
            )
            all_uv.extend([(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
            all_segA.extend([(ax_, ay_)] * 4)
            all_segB.extend([(bx_, by_)] * 4)
            indices.append((base, base + 1, base + 2))
            indices.append((base + 2, base + 1, base + 3))
            base_vert += 4

        _emit_cap(centers[n][0], centers[n][1])

    batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": all_pos, "uv": all_uv, "segA": all_segA, "segB": all_segB},
        indices=indices,
    )
    return shader, batch


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


def _draw_filled_rounded_rect(x, y, width, height, radius, color):
    if width <= 0 or height <= 0:
        return
    radius = max(0, min(radius, width / 2, height / 2))

    shader = _get_sdf_fill_shader()
    half_w, half_h = width / 2, height / 2

    vertices = (
        (x, y, 0.0),
        (x + width, y, 0.0),
        (x + width, y + height, 0.0),
        (x, y + height, 0.0),
    )
    uvs = (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (half_w, half_h))
    shader.uniform_float("radius", radius)
    batch.draw(shader)


def _draw_filled_rounded_rect_varying(x, y, width, height, radii, color):
    if width <= 0 or height <= 0:
        return
    max_r = min(width / 2, height / 2)
    radii = tuple(max(0, min(r, max_r)) for r in radii)

    shader = _get_sdf_fill_varying_shader()
    half_w, half_h = width / 2, height / 2

    vertices = (
        (x, y, 0.0),
        (x + width, y, 0.0),
        (x + width, y + height, 0.0),
        (x, y + height, 0.0),
    )
    uvs = (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (half_w, half_h))
    shader.uniform_float("radii", radii)
    batch.draw(shader)


def _draw_filled_rounded_rect_with_hole(
    map_x,
    map_y,
    map_w,
    map_h,
    outer_radius,
    inner_x,
    inner_y,
    inner_w,
    inner_h,
    inner_radius,
    color,
):
    if map_w <= 0 or map_h <= 0 or inner_w <= 0 or inner_h <= 0:
        return
    outer_radius = max(0, min(outer_radius, map_w / 2, map_h / 2))

    shader = _get_sdf_fill_hole_shader()
    half_w, half_h = map_w / 2, map_h / 2

    inner_off_x = (inner_x + inner_w / 2) - (map_x + map_w / 2)
    inner_off_y = (inner_y + inner_h / 2) - (map_y + map_h / 2)
    inner_half_w = inner_w / 2
    inner_half_h = inner_h / 2

    vertices = (
        (map_x, map_y, 0.0),
        (map_x + map_w, map_y, 0.0),
        (map_x + map_w, map_y + map_h, 0.0),
        (map_x, map_y + map_h, 0.0),
    )
    uvs = (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("outerData", (half_w, half_h, outer_radius, inner_radius))
    shader.uniform_float("innerOffset", (inner_off_x, inner_off_y))
    shader.uniform_float("innerHalfSize", (inner_half_w, inner_half_h))
    batch.draw(shader)


def _draw_filled_rounded_rect_clipped(x, y, width, height, radius, color, clip_x, clip_y, clip_w, clip_h, clip_radius):
    """Draw a rounded rect fill intersected with a rounded clip region."""
    if width <= 0 or height <= 0 or clip_w <= 0 or clip_h <= 0:
        return
    radius = max(0, min(radius, width / 2, height / 2))
    clip_radius = max(0, min(clip_radius, clip_w / 2, clip_h / 2))

    shader = _get_sdf_fill_clip_shader()
    half_w, half_h = width / 2, height / 2
    center_x, center_y = x + half_w, y + half_h
    clip_half_w, clip_half_h = clip_w / 2, clip_h / 2
    off_x = (clip_x + clip_half_w) - center_x
    off_y = (clip_y + clip_half_h) - center_y

    # Pad the quad so the clip arc's AA falloff has room at shared edges
    aa_pad = 2.0
    corner_x0, corner_y0 = x - aa_pad, y - aa_pad
    corner_x1, corner_y1 = x + width + aa_pad, y + height + aa_pad

    vertices = (
        (corner_x0, corner_y0, 0.0),
        (corner_x1, corner_y0, 0.0),
        (corner_x1, corner_y1, 0.0),
        (corner_x0, corner_y1, 0.0),
    )
    uvs = (
        (corner_x0 - center_x, corner_y0 - center_y),
        (corner_x1 - center_x, corner_y0 - center_y),
        (corner_x1 - center_x, corner_y1 - center_y),
        (corner_x0 - center_x, corner_y1 - center_y),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("sizeData", (half_w, half_h, radius, clip_radius))
    shader.uniform_float("clipData", (off_x, off_y, clip_half_w, clip_half_h))
    batch.draw(shader)


def _draw_rounded_rect_border(x, y, width, height, radius, color, line_width=1.0):
    if width <= 0 or height <= 0:
        return
    radius = max(0, min(radius, width / 2, height / 2))

    shader = _get_sdf_border_shader()
    half_w, half_h = width / 2, height / 2

    vertices = (
        (x, y, 0.0),
        (x + width, y, 0.0),
        (x + width, y + height, 0.0),
        (x, y + height, 0.0),
    )
    uvs = (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (half_w, half_h))
    shader.uniform_float("radius", radius)
    shader.uniform_float("lineWidth", line_width)
    batch.draw(shader)


def _draw_rounded_rect_border_varying_sides(
    x, y, width, height, radii, color, line_width=1.0, skip_left=False, skip_right=False
):
    if width <= 0 or height <= 0:
        return
    max_r = min(width / 2, height / 2)
    radii = tuple(max(0, min(r, max_r)) for r in radii)

    shader = _get_sdf_border_varying_sides_shader()
    half_w, half_h = width / 2, height / 2

    vertices = (
        (x, y, 0.0),
        (x + width, y, 0.0),
        (x + width, y + height, 0.0),
        (x, y + height, 0.0),
    )
    uvs = (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    )
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (half_w, half_h))
    shader.uniform_float("radii", radii)
    shader.uniform_float(
        "lineData",
        (line_width, 1.0 if skip_left else 0.0, 1.0 if skip_right else 0.0, 0.0),
    )
    batch.draw(shader)


def _draw_pill(x, y, width, height, color):
    if width <= 0 or height <= 0:
        return

    shader = _get_pill_shader()
    half_w, half_h = width / 2, height / 2

    aa_pad = 2.0

    vertices = (
        (x - aa_pad, y - aa_pad, 0.0),
        (x + width + aa_pad, y - aa_pad, 0.0),
        (x + width + aa_pad, y + height + aa_pad, 0.0),
        (x - aa_pad, y + height + aa_pad, 0.0),
    )
    uvs = (
        (-half_w - aa_pad, -half_h - aa_pad),
        (half_w + aa_pad, -half_h - aa_pad),
        (half_w + aa_pad, half_h + aa_pad),
        (-half_w - aa_pad, half_h + aa_pad),
    )

    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )

    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (half_w, half_h))

    batch.draw(shader)


def _draw_pill_border(x, y, width, height, color, line_width=1.0):
    if width <= 0 or height <= 0:
        return

    shader = _get_pill_border_shader()
    half_w, half_h = width / 2, height / 2

    aa_pad = 2.0

    vertices = (
        (x - aa_pad, y - aa_pad, 0.0),
        (x + width + aa_pad, y - aa_pad, 0.0),
        (x + width + aa_pad, y + height + aa_pad, 0.0),
        (x - aa_pad, y + height + aa_pad, 0.0),
    )
    uvs = (
        (-half_w - aa_pad, -half_h - aa_pad),
        (half_w + aa_pad, -half_h - aa_pad),
        (half_w + aa_pad, half_h + aa_pad),
        (-half_w - aa_pad, half_h + aa_pad),
    )

    batch = batch_for_shader(shader, "TRIS", {"pos": vertices, "uv": uvs}, indices=((0, 1, 2), (2, 3, 0)))

    shader.bind()
    shader.uniform_float(
        "ModelViewProjectionMatrix",
        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
    )
    shader.uniform_float("color", _srgb_to_linear(color))
    shader.uniform_float("halfSize", (half_w, half_h))
    shader.uniform_float("lineWidth", line_width)

    batch.draw(shader)
