#version 460 core

layout (location = 0) in vec2 uv;
layout (location = 1) in flat uint layer;

layout (location = 0) out vec4 pixel;

layout (location = 0) uniform sampler2DArray attachment;

void main() {
    pixel = vec4(vec3(texture(attachment, vec3(uv, layer)).r), 1.0);
}
