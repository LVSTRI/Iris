#version 460 core

layout (location = 0) in vec2 i_uv;

layout (location = 0) out vec4 o_pixel;

layout (location = 0) uniform sampler2D u_attachment;

void main() {
    o_pixel = vec4(texture(u_attachment, i_uv).rgb, 1.0);
}
