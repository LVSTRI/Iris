#version 460 core

layout (location = 0) out vec4 pixel;

// even though we only supplied 4 colors per vertex, the rasterizer will interpolate the values based on the generated fragments.
layout (location = 0) in vec3 color;
layout (location = 1) in vec2 uvs;

layout (location = 0) uniform sampler2D tex;

void main() {
    pixel = texture(tex, uvs) * vec4(color, 1.0);
}
