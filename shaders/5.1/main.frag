#version 460 core
#extension GL_ARB_bindless_texture : enable

#define M_GOLDEN_CONJ 0.6180339887498948482045868343656

layout (early_fragment_tests) in;

struct indirect_command_t {
    uint count;
    uint instance_count;
    uint first_index;
    int base_vertex;
    uint base_instance;
};

struct aabb_t {
    vec4 min;
    vec4 max;
    vec4 center;
    vec4 size;
};

struct object_info_t {
    uint local_transform;
    uint global_transform;
    uint diffuse_texture;
    uint normal_texture;
    uint specular_texture;
    uint group_index;
    uint group_offset;

    vec4 sphere; // w is radius
    aabb_t aabb;
    indirect_command_t command;
};

layout (location = 0) in t_per_vertex {
    flat uint meshlet_id;
    flat uint mesh_id;
    vec2 uv;
} o_per_vertex;

layout (location = 0) out vec4 o_pixel;

layout (std430, binding = 5) readonly restrict buffer b_object_info {
    object_info_t[] objects;
};

layout (std430, binding = 6) readonly restrict buffer b_textures {
    sampler2D[] textures;
};

vec3 hsv_to_rgb(in vec3 hsv) {
    const vec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsv.z * mix(vec3(1.0), rgb, hsv.y);
}

void main() {
    /*const uint diffuse_index = objects[o_per_vertex.mesh_id].diffuse_texture;
    vec3 diffuse = vec3(1.0);
    if (diffuse_index != -1) {
        diffuse = texture(textures[diffuse_index], o_per_vertex.uv).rgb;
    }
    o_pixel = vec4(diffuse, 1.0);*/
    o_pixel = vec4(hsv_to_rgb(vec3(fract(M_GOLDEN_CONJ * (o_per_vertex.meshlet_id + 1)), 0.875, 0.85)), 1.0);
}
