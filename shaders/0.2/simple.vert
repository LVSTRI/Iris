#version 460 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec2 uvs;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uvs;

void main() {
    gl_Position = vec4(position, 1.0);
    out_color = color;
    out_uvs = uvs;
}
