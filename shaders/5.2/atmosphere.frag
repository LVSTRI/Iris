#version 460 core

#define M_PI 3.1415926535897932384626433832795
#define M_GOLDEN 1.6180339887498948482045868343656
#define M_GOLDEN_CONJ 0.6180339887498948482045868343656

struct directional_light_t {
    vec3 direction;
    vec3 diffuse;
    vec3 specular;
};

layout (location = 0) in vec2 i_uv;

layout (location = 0) out vec4 o_pixel;

layout (location = 0) uniform vec2 u_resolution;
layout (location = 1) uniform float u_sun_angular_measure;

layout (std140, binding = 0) uniform u_camera {
    mat4 inf_projection;
    mat4 projection;
    mat4 view;
    mat4 pv;
    vec3 position;
    float near;
    float far;
} camera;

layout (std140, binding = 1) uniform b_directional_lights {
    directional_light_t[4] directional_lights;
};

vec3 as_srgb(in vec3 color) {
    vec3 o = vec3(0.0);
    for (uint i = 0; i < 3; ++i) {
        if (color[i] < 0.0031308) {
            o[i] = 12.92 * color[i];
        } else {
            o[i] = 1.055 * pow(color[i], 1.0 / 2.4) - 0.055;
        }
    }
    return o;
}

vec3 calculate_sun(in directional_light_t sun) {
    const vec2 ndc = i_uv * 2.0 - 1.0;
    const vec3 ndc_near = vec3(ndc, 0.0);
    const vec3 ndc_far = vec3(ndc, 1.0);
    vec4 world_near = inverse(camera.pv) * vec4(ndc_near, 1.0);
    vec4 world_far = inverse(camera.pv) * vec4(ndc_far, 1.0);
    world_near /= world_near.w;
    world_far /= world_far.w;
    const vec3 orig = world_near.xyz;
    const vec3 dir = normalize(world_far.xyz - world_near.xyz);
    const float dir_up = dot(dir, vec3(0.0, 1.0, 0.0));
    const float dir_sun = dot(dir, -sun.direction);
    if (dir_sun < cos(u_sun_angular_measure)) {
        return sun.diffuse;
    }
    return normalize(vec3(0.527, 0.804, 0.917)) * clamp(dir_up * 0.7, 0.3, 1.0);
}

void main() {
    const vec3 sun = calculate_sun(directional_lights[0]);
    o_pixel = vec4(as_srgb(sun), 1.0);
}