#version 460 core

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 out_pixel;

layout (location = 0) uniform sampler2D color[2];
layout (location = 2) uniform uint frame;

void main() {
    const vec3 old = texture(color[0], uv).rgb;
    const vec3 new = texture(color[1], uv).rgb;

    const float weight = 1.0 / float(frame + 1);
    out_pixel = vec4(mix(old, new, weight), 1.0);
}
