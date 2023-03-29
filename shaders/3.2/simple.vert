#version 460 core

#define MAX_INSTANCES 256

struct transform_data_t {
    mat4 model;
    mat4 t_inv_model;
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 out_frag_pos;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
} camera;

layout (std140, binding = 1) uniform transform_uniform_t {
    transform_data_t[MAX_INSTANCES] data;
} transform;

layout (location = 0) uniform uint transform_id;

void main() {
    const vec4 frag_pos = transform.data[transform_id].model * vec4(position, 1.0);
    out_frag_pos = vec3(frag_pos);
    out_normal = normalize(mat3(transform.data[transform_id].t_inv_model) * normal);
    out_uv = uv;

    gl_Position = camera.projection * camera.view * frag_pos;
}
