#version 460 core
#extension GL_ARB_bindless_texture : enable

layout (early_fragment_tests) in;

layout (location = 0) in flat uint i_diffuse_texture;
layout (location = 1) in vec2 i_uv;

layout (std430, binding = 5) readonly restrict buffer b_textures {
    sampler2D[] textures;
};

void main() {
    if (i_diffuse_texture != -1 && texture(textures[i_diffuse_texture], i_uv).a < 0.5) {
        discard;
    }
}
