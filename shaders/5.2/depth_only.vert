#version 460 core

invariant gl_Position;

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

    vec4 scale;
    vec4 sphere; // w is radius
    aabb_t aabb;
    indirect_command_t command;
};

struct object_index_shift_t {
    uint object_id;
};

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;
layout (location = 2) in vec2 i_uv;
layout (location = 3) in vec4 i_tangent;

layout (location = 0) uniform uint u_group_offset;
layout (location = 1) uniform vec2 u_jitter;

layout (std140, binding = 0) uniform u_camera {
    mat4 inf_projection;
    mat4 projection;
    mat4 view;
    mat4 pv;
    vec3 position;
    float near;
    float far;
} camera;

layout (std430, binding = 1) readonly restrict buffer b_local_transform {
    mat4[] local_transforms;
};

layout (std430, binding = 2) readonly restrict buffer b_global_transform {
    mat4[] global_transforms;
};

layout (std430, binding = 3) readonly restrict buffer b_object_info {
    object_info_t[] objects;
};

layout (std430, binding = 4) restrict buffer b_object_index_shift {
    object_index_shift_t[] object_shift;
};

void main() {
    const object_info_t object_info = objects[object_shift[gl_DrawID + u_group_offset].object_id];
    const mat4 global_transform = global_transforms[object_info.global_transform];
    const mat4 local_transform = local_transforms[object_info.local_transform];
    const mat4 transform = global_transform * local_transform;
    const vec4 clip_pos = camera.pv * transform * vec4(i_position, 1.0);
    gl_Position = clip_pos + vec4(u_jitter * clip_pos.w, 0.0, 0.0);
}