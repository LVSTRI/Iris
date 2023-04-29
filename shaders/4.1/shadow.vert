#version 460 core
#define CASCADE_COUNT 4

struct transform_data_t {
    mat4 model;
    mat4 t_inv_model;
};

struct cascade_data_t {
    mat4 projection;
    mat4 view;
    mat4 pv;
    mat4 global;
    vec4 scale;
    vec4 offset; // w is split
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (std430, binding = 0) readonly restrict buffer cascade_output_t {
    cascade_data_t[CASCADE_COUNT] cascades;
};

layout (std140, binding = 1) readonly restrict buffer transform_buffer_t {
    transform_data_t[] data;
} transform;

layout (location = 0) uniform uint transform_id;
layout (location = 1) uniform uint layer;

void main() {
    gl_Position = cascades[layer].pv * transform.data[transform_id].model * vec4(position, 1.0);
    // gl_Layer = int(layer);
}
