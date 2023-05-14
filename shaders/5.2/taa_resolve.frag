#version 460 core

layout (location = 0) in vec2 i_uv;

layout (location = 0) out vec4 o_pixel;

layout (location = 0) uniform sampler2D current_color;
layout (location = 1) uniform sampler2D history_color;
layout (location = 2) uniform sampler2D velocity_vectors;

void main() {
    const vec2 velocity = textureLod(velocity_vectors, i_uv, 0.0).rg;
    const vec2 prev_uv = i_uv - velocity;
    const vec3 current = textureLod(current_color, i_uv, 0.0).rgb;
    const vec3 history = textureLod(history_color, prev_uv, 0.0).rgb;

    const vec3[] color_samples = vec3[](
        textureLodOffset(current_color, i_uv, 0.0, ivec2( 1,  0)).rgb,
        textureLodOffset(current_color, i_uv, 0.0, ivec2( 0,  1)).rgb,
        textureLodOffset(current_color, i_uv, 0.0, ivec2(-1,  0)).rgb,
        textureLodOffset(current_color, i_uv, 0.0, ivec2( 0, -1)).rgb);
    const vec3 color_max = max(current, max(color_samples[0], max(color_samples[1], max(color_samples[2], color_samples[3]))));
    const vec3 color_min = min(current, min(color_samples[0], min(color_samples[1], min(color_samples[2], color_samples[3]))));
    const vec3 history_clamped = clamp(history, color_min, color_max);
    o_pixel = vec4(mix(current, history_clamped, 0.9), 1.0);
}
