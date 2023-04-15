#version 460 core

struct transform_data_t {
    mat4 model;
    mat4 t_inv_model;
};

struct shadow_frustum_t {
    mat4 projection;
    mat4 view;
    vec4 partitions;
};

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (std430, binding = 0) readonly restrict buffer shadow_frustums_t {
    shadow_frustum_t[] shadow_frustums;
};

layout (std140, binding = 1) readonly restrict buffer transform_buffer_t {
    transform_data_t[] data;
} transform;

layout (location = 0) uniform uint transform_id;
layout (location = 1) uniform uint layer;

void main() {
    gl_Position = shadow_frustums[layer].projection * shadow_frustums[layer].view * transform.data[transform_id].model * vec4(position, 1.0);
    // gl_Layer = int(layer);
}
