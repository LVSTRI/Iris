#version 460 core

layout (location = 0) out vec4 out_pixel;
layout (location = 1) out uint out_id;

layout (location = 3) uniform vec3 light_color;

void main() {
    out_pixel = vec4(light_color, 1.0);
    out_id = 0xffffffff;
}
