#version 460 core

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

layout (location = 0) uniform mat4 model;

void main() {
    gl_Position = camera.projection * camera.view * model * vec4(position, 1.0);
}