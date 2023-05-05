#version 460 core
#define CASCADE_COUNT 4

struct cascade_data_t {
    mat4 projection;
    mat4 view;
    mat4 pv;
    mat4 global;
    vec4 scale;
    vec4 offset; // w is split
};

struct indirect_command_t {
    uint count;
    uint instance_count;
    uint first_index;
    int base_vertex;
    uint base_instance;
};

struct object_info_t {
    uint local_transform;
    uint global_transform;
    uint diffuse_texture;
    uint normal_texture;
    uint specular_texture;
    uint group_index;
    uint group_offset;

    indirect_command_t command;
};

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;
layout (location = 2) in vec2 i_uv;
layout (location = 3) in vec4 i_tangent;

layout (location = 0) out flat uint o_diffuse_texture;
layout (location = 1) out vec2 o_uv;

layout (location = 0) uniform uint layer;
layout (location = 1) uniform uint object_offset;

layout (std430, binding = 0) readonly restrict buffer b_cascade_output {
    cascade_data_t[CASCADE_COUNT] cascades;
};

layout (std430, binding = 1) readonly restrict buffer b_local_transform {
    mat4[] local_transforms;
};

layout (std430, binding = 2) readonly restrict buffer b_global_transform {
    mat4[] global_transforms;
};

layout (std430, binding = 3) readonly restrict buffer b_object_info {
    object_info_t[] objects;
};

void main() {
    const object_info_t object_info = objects[object_offset + gl_DrawID];
    const mat4 global_transform = global_transforms[object_info.global_transform];
    const mat4 local_transform = local_transforms[object_info.local_transform];
    const mat4 transform = global_transform * local_transform;
    gl_Position = cascades[layer].pv * transform * vec4(i_position, 1.0);
    o_diffuse_texture = object_info.diffuse_texture;
    o_uv = i_uv;
}
