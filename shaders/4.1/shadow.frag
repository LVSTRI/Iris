#version 460 core

layout (location = 0) in vec2 uv;

layout (location = 2) uniform sampler2D diffuse;

void main() {
    /*if (texture(diffuse, uv).a < 0.5) {
        discard;
    }*/
}
