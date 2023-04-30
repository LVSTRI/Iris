#version 460 core

invariant gl_Position;

struct transform_data_t {
    mat4 model;
    mat4 t_inv_model;
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
    vec3 position;
    float near;
    float far;
} camera;

layout (std140, binding = 1) readonly restrict buffer transform_buffer_t {
    transform_data_t[] data;
} transform;

layout (location = 0) uniform uint transform_id;

void main() {
    const vec4 frag_pos = transform.data[transform_id].model * vec4(position, 1.0);
    gl_Position = camera.projection * camera.view * frag_pos;
}
