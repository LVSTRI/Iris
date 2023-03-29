#version 460 core

#define MAX_INSTANCES 1024

struct transform_data_t {
    mat4 model;
    mat4 t_inv_model;
};

layout (location = 0) in vec3 position;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
} camera;

layout (location = 0) uniform mat4 model;

void main() {
    gl_Position = camera.projection * camera.view * model * vec4(position, 1.0);
}
