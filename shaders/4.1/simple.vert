#version 460 core

struct transform_data_t {
    mat4 model;
    mat4 t_inv_model;
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 out_frag_pos;
layout (location = 1) out vec4 out_frag_pos_shadow;
layout (location = 2) out vec3 out_normal;
layout (location = 3) out vec2 out_uv;
layout (location = 4) out flat vec3 out_camera_pos;
layout (location = 5) out flat uint out_transform_id;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (std140, binding = 1) uniform shadow_camera_uniform_t {
    mat4 projection;
    mat4 view;
    vec3 position;
} shadow_camera;

layout (std140, binding = 2) readonly restrict buffer transform_buffer_t {
    transform_data_t[] data;
} transform;

layout (location = 0) uniform uint transform_id;

void main() {
    const vec4 frag_pos = transform.data[transform_id].model * vec4(position, 1.0);
    out_frag_pos = vec3(frag_pos);
    out_frag_pos_shadow = shadow_camera.projection * shadow_camera.view * frag_pos;
    out_normal = normalize(mat3(transform.data[transform_id].t_inv_model) * normal);
    out_uv = uv;
    out_camera_pos = camera.position;
    out_transform_id = transform_id;

    gl_Position = camera.projection * camera.view * frag_pos;
}
