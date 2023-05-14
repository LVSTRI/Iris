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

layout (location = 0) out flat uint o_object_id;

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

layout (std430, binding = 5) writeonly restrict buffer b_roc_output {
    uint[] roc_visibility;
};

vec3 make_cube(in uint vertex) {
    const uint b = 1 << vertex;
    return vec3(
        (0x287au & b) != 0,
        (0x02afu & b) != 0,
        (0x31e3u & b) != 0);
}


void main() {
    const uint object_id = object_shift[gl_InstanceID].object_id;
    const object_info_t object_info = objects[object_id];
    const mat4 global_transform = global_transforms[object_info.global_transform];
    const mat4 local_transform = local_transforms[object_info.local_transform];
    const mat4 transform = global_transform * local_transform;

    vec3 position = make_cube(gl_VertexID) - 0.5;
    const vec3 aabb_max = object_info.aabb.max.xyz;
    const vec3 aabb_min = object_info.aabb.min.xyz;
    const vec3 aabb_size = (aabb_max - aabb_min) * 0.5;
    const vec3 aabb_center = object_info.aabb.center.xyz;
    const vec3 local_view_pos = vec3(transpose(inverse(transform)) * vec4(camera.position, 1.0));
    position *= aabb_size * 2.0 + 20;
    position += aabb_center;
    o_object_id = object_id;
    if (all(lessThan(abs(local_view_pos - aabb_center), aabb_size + 20))) {
        roc_visibility[object_id] = 1;
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
    } else {
        gl_Position = camera.pv * transform * vec4(position, 1.0);
    }
}
