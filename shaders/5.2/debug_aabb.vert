#version 460 core

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

mat4 mat4_scale_translate(in vec3 t, in vec3 s) {
    return mat4(
        vec4(s.x, 0.0, 0.0, 0.0),
        vec4(0.0, s.y, 0.0, 0.0),
        vec4(0.0, 0.0, s.z, 0.0),
        vec4(t.x, t.y, t.z, 1.0));
}

void main() {
    const object_info_t object_info = objects[object_shift[gl_InstanceID + gl_BaseInstance].object_id];
    const mat4 global_transform = global_transforms[object_info.global_transform];
    const mat4 local_transform = local_transforms[object_info.local_transform];
    const mat4 transform = global_transform * local_transform;

    const vec3 aabb_center = object_info.aabb.center.xyz;
    const vec3 aabb_size = object_info.aabb.max.xyz - object_info.aabb.min.xyz;
    const mat4 aabb_transform = mat4_scale_translate(aabb_center, aabb_size / 2.0);
    gl_Position = camera.pv * transform * aabb_transform * vec4(i_position, 1.0);
}
