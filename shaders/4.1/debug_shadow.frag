#version 460 core

layout (location = 0) in vec2 uv;

layout (location = 0) out vec4 pixel;

layout (location = 0) uniform sampler2D attachment;

void main() {
    pixel = vec4(vec3(texture(attachment, uv).r), 1.0);
}
