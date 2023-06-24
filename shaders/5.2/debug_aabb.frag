#version 460 core

layout (location = 0) out vec4 o_pixel;
layout (location = 1) out vec2 o_velocity;

void main() {
    o_pixel = vec4(1.0);
    o_velocity = vec2(0.0);
}
