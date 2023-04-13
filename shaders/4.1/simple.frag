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

layout (location = 0) in vec3 frag_pos;
layout (location = 1) in vec4 frag_pos_shadow;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;
layout (location = 4) in flat vec3 camera_pos;
layout (location = 5) in flat uint transform_id;

layout (location = 0) out vec4 out_pixel;
layout (location = 1) out uint out_transform_id;

layout (location = 4) uniform material_t material;
layout (location = 7) uniform uint n_point_lights;
layout (location = 8) uniform sampler2D shadow_map;

layout (std140, binding = 3) readonly restrict buffer point_light_uniform_t {
    point_light_t[] data;
} point_light;

layout (std140, binding = 4) uniform directional_light_uniform_t {
    directional_light_t dir_light;
};

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

float calculate_shadow() {
    const vec3 proj_coords = frag_pos_shadow.xyz * 0.5 + 0.5;
    if (proj_coords.z > 1.0) {
        return 1.0;
    }
    const vec2 texel_size = 1.0 / textureSize(shadow_map, 0).xy;
    const vec2[] offsets = vec2[](
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
    const float bias = 0.0125;
    float sampled_depth = 0.0;
    for (uint i = 0; i < 16; ++i) {
        const float closest = texture(shadow_map, proj_coords.xy + offsets[i] * texel_size).r;
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

    vec3 color = ambient;
    for (uint i = 0; i < n_point_lights; i++) {
        color += calculate_point_light(point_light.data[i], diffuse, specular, n_normal);
    }

    color +=
        calculate_directional_light(diffuse, specular, n_normal) *
        calculate_shadow();

    out_pixel = vec4(color, alpha);
    out_transform_id = transform_id;
}
