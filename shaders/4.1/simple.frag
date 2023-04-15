#version 460 core

struct material_t {
    sampler2D diffuse;
    sampler2D specular;
    uint shininess;
};

struct directional_light_t {
    vec3 direction;
    vec3 diffuse;
    vec3 specular;
};

struct point_light_t {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

struct shadow_frustum_t {
    mat4 projection;
    mat4 view;
    vec4 partitions;
};

layout (location = 0) in vec3 frag_pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;
layout (location = 3) in flat vec3 camera_pos;
layout (location = 4) in flat uint transform_id;

layout (location = 0) out vec4 out_pixel;
layout (location = 1) out uint out_transform_id;

layout (location = 1) uniform uint num_cascades;
layout (location = 4) uniform material_t material;
layout (location = 7) uniform uint n_point_lights;
layout (location = 8) uniform sampler2DArray shadow_map;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (std430, binding = 2) readonly restrict buffer shadow_camera_buffer_t {
    shadow_frustum_t[] shadow_frustums;
};

layout (std140, binding = 3) readonly restrict buffer point_light_uniform_t {
    point_light_t[] data;
} point_light;

layout (std140, binding = 4) uniform directional_light_uniform_t {
    directional_light_t dir_light;
};

uint calculate_cascade() {
    const float dist = abs((camera.view * vec4(frag_pos, 1.0)).z);
    for (uint i = 0; i < num_cascades; ++i) {
        if (dist < shadow_frustums[i].partitions[1]) {
            return i;
        }
    }
    return num_cascades - 1;
}

vec3 calculate_point_light(point_light_t light, vec3 diffuse_color, vec3 specular_color, vec3 normal) {
    const vec3 ambient_result = light.ambient * diffuse_color;

    const vec3 light_dir = normalize(light.position - frag_pos);
    const float diffuse_intensity = max(dot(light_dir, normal), 0.0);
    const vec3 diffuse_result = light.diffuse * diffuse_intensity * diffuse_color;

    const vec3 view_dir = normalize(camera_pos - frag_pos);
    const vec3 halfway = normalize(light_dir + view_dir);
    const float specular_intensity = pow(max(dot(normal, halfway), 0.0), material.shininess);
    const vec3 specular_result = light.specular * specular_intensity * specular_color;

    const float f_dist = length(light.position - frag_pos);
    const float f_dist2 = f_dist * f_dist;
    const float attenuation =
        1.0 / (
            light.constant +
            f_dist * light.linear +
            f_dist2 * light.quadratic);

    return attenuation * (ambient_result + diffuse_result + specular_result);
}

vec3 calculate_directional_light(vec3 diffuse_color, vec3 specular_color, vec3 normal) {
    const directional_light_t light = dir_light;

    const vec3 light_dir = normalize(light.direction);
    const float diffuse_intensity = max(dot(light_dir, normal), 0.0);
    const vec3 diffuse_result = light.diffuse * diffuse_intensity * diffuse_color;

    const vec3 view_dir = normalize(camera_pos - frag_pos);
    const vec3 halfway = normalize(light_dir + view_dir);
    const float specular_intensity = pow(max(dot(normal, halfway), 0.0), material.shininess);
    const vec3 specular_result = light.specular * specular_intensity * specular_color;

    return diffuse_result + specular_result;
}

float calculate_shadow(vec3 normal, uint cascade) {
    const vec3 frag_pos_shadow = vec3(shadow_frustums[cascade].projection * shadow_frustums[cascade].view * vec4(frag_pos, 1.0));
    const vec3 proj_coords = frag_pos_shadow.xyz * 0.5 + 0.5;
    if (proj_coords.z > 1.0) {
        return 1.0;
    }
    const vec2 texel_size = 1.0 / textureSize(shadow_map, 0).xy;
    const vec2[] shadow_sampling_offsets = vec2[](
        vec2(0.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.5, 0.0),
        vec2(0.0, 0.5),
        vec2(0.5, 0.5),
        vec2(0.5, 1.0),
        vec2(1.0, 0.5),
        vec2(0.25, 0.25),
        vec2(0.25, 0.75),
        vec2(0.75, 0.25),
        vec2(0.75, 0.75),
        vec2(0.25, 0.5),
        vec2(0.5, 0.25),
        vec2(0.75, 0.5));
    const vec3 n_light_dir = normalize(dir_light.direction);
    const float width = 0.00025;
    const float bias = clamp((width / 2.0) * tan(acos(clamp(dot(normal, n_light_dir), -1.0, 1.0))), 0.0, width);
    float sampled_depth = 0.0;
    for (uint i = 0; i < 16; ++i) {
        const vec2 s_uv = proj_coords.xy + shadow_sampling_offsets[i] * texel_size;
        const float closest = texture(shadow_map, vec3(s_uv, cascade)).r;
        const float current = proj_coords.z;
        sampled_depth += 1.0 - float(current - bias > closest);
    }
    return sampled_depth / 16.0;
}

void main() {
    const float ambient_factor = 0.025;
    const vec3 diffuse = texture(material.diffuse, uv).rgb;
    const float alpha = texture(material.diffuse, uv).a;
    const vec3 specular = texture(material.specular, uv).rgb;
    const vec3 ambient = diffuse * ambient_factor;
    const vec3 n_normal = normalize(normal);
    const uint cascade = calculate_cascade();

    vec3 color = ambient;
    for (uint i = 0; i < n_point_lights; i++) {
        color += calculate_point_light(point_light.data[i], diffuse, specular, n_normal);
    }

    color +=
        calculate_directional_light(diffuse, specular, n_normal) *
        calculate_shadow(n_normal, cascade);

    out_pixel = vec4(color, alpha);
    out_transform_id = transform_id;
}
