#version 460 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec2 out_uv;
layout (location = 1) out flat uint out_layer;

layout (location = 1) uniform uint layer;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    out_layer = layer;
    out_uv = uv;
}
