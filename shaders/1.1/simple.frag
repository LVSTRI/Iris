#version 460 core

#define MAX_POINT_LIGHTS 4

struct material_t {
    sampler2D diffuse;
    sampler2D specular;
    uint shininess;
};

struct directional_light_t {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct attenuation_t {
    float constant;
    float linear;
    float quadratic;
};

struct point_light_t {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    attenuation_t attenuation;
};

layout (location = 0) in vec3 frag_pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec4 pixel;

layout (location = 4) uniform vec3 camera_pos;
layout (location = 5) uniform material_t material;
layout (location = 8) uniform point_light_t point_lights[MAX_POINT_LIGHTS];
layout (location = 36) uniform directional_light_t dir_light;


vec3 calculate_directional_light() {
    const vec3 diffuse_color = vec3(texture(material.diffuse, uv));
    const vec3 specular_color = vec3(texture(material.specular, uv));
    const vec3 ambient_result = dir_light.ambient * diffuse_color;

    const vec3 light_dir = normalize(dir_light.direction);
    const float diffuse_intensity = max(dot(light_dir, normal), 0.0);
    const vec3 diffuse_result = diffuse_color * diffuse_intensity * dir_light.diffuse;

    const vec3 view_dir = normalize(camera_pos - frag_pos);
    const vec3 specular_dir = reflect(-light_dir, normal);
    const float specular_intensity = pow(max(dot(view_dir, specular_dir), 0.0), material.shininess);
    const vec3 specular_result = specular_color * specular_intensity * dir_light.specular;

    return ambient_result + diffuse_result + specular_result;
}

vec3 calculate_point_light(point_light_t point_light) {
    const vec3 diffuse_color = vec3(texture(material.diffuse, uv));
    const vec3 specular_color = vec3(texture(material.specular, uv));
    const vec3 ambient_result = point_light.ambient * diffuse_color;

    const vec3 light_dir = normalize(point_light.position - frag_pos);
    const float diffuse_intensity = max(dot(light_dir, normal), 0.0);
    const vec3 diffuse_result = diffuse_color * diffuse_intensity * point_light.diffuse;

    const vec3 view_dir = normalize(camera_pos - frag_pos);
    const vec3 specular_dir = reflect(-light_dir, normal);
    const float specular_intensity = pow(max(dot(view_dir, specular_dir), 0.0), material.shininess);
    const vec3 specular_result = specular_color * specular_intensity * point_light.specular;

    const float distance = length(point_light.position - frag_pos);
    const float distance_2 = distance * distance;
    const float attenuation =
        1.0 / (point_light.attenuation.constant +
               point_light.attenuation.linear * distance +
               point_light.attenuation.quadratic * distance_2);
    return attenuation * (ambient_result + diffuse_result + specular_result);
}

void main() {
    const vec3 dir_light_color = calculate_directional_light();

    vec3 color = dir_light_color;
    for (uint i = 0; i < MAX_POINT_LIGHTS; ++i) {
        color += calculate_point_light(point_lights[i]);
    }

    pixel = vec4(color, 1.0);
}
