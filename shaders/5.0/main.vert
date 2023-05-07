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

layout (location = 0) out flat uint o_diffuse_texture;
layout (location = 1) out flat uint o_normal_texture;
layout (location = 2) out flat uint o_specular_texture;
layout (location = 3) out flat uint o_draw_id;
layout (location = 4) out vec3 o_normal;
layout (location = 5) out vec2 o_uv;
layout (location = 6) out vec3 o_frag_pos;
layout (location = 7) out mat3 o_TBN;

layout (location = 0) uniform uint group_offset;

layout (std140, binding = 0) uniform u_camera {
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

mat3 mat3_make_tbn(in mat3 transform) {
    const vec3 bitangent = cross(i_normal, i_tangent.xyz) * i_tangent.w;
    const vec3 T = normalize(transform * i_tangent.xyz);
    const vec3 B = normalize(transform * bitangent);
    const vec3 N = normalize(transform * i_normal);
    return mat3(T, B, N);
}

void main() {
    const object_info_t object_info = objects[object_shift[gl_DrawID + group_offset].object_id];
    const mat4 global_transform = global_transforms[object_info.global_transform];
    const mat4 local_transform = local_transforms[object_info.local_transform];
    const mat4 transform = global_transform * local_transform;
    const mat4 inv_transform = inverse(transpose(transform));
    const mat3 TBN = mat3_make_tbn(mat3(transform));
    o_diffuse_texture = object_info.diffuse_texture;
    o_normal_texture = object_info.normal_texture;
    o_specular_texture = object_info.specular_texture;
    o_normal = normalize(mat3(inv_transform) * i_normal);
    o_uv = i_uv;
    o_draw_id = group_offset + gl_DrawID;
    o_frag_pos = vec3(transform * vec4(i_position, 1.0));
    o_TBN = TBN;
    gl_Position = camera.pv * vec4(o_frag_pos, 1.0);
}
