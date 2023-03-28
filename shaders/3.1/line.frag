#version 460 core

layout (location = 0) out vec4 pixel;

layout (location = 3) uniform vec3 color;

void main() {
    pixel = vec4(color, 1.0);
}
