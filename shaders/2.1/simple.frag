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
layout (location = 8) uniform point_light_t[MAX_POINT_LIGHTS] point_lights;

vec3 calculate_point_light(point_light_t light, vec3 diffuse_color, vec3 specular_color) {
    const vec3 ambient_result = light.ambient * diffuse_color;

    const vec3 light_dir = normalize(light.position - frag_pos);
    const float diffuse_intensity = max(dot(light_dir, normal), 0.0);
    const vec3 diffuse_result = light.diffuse * diffuse_intensity * diffuse_color;

    const vec3 view_dir = normalize(camera_pos - frag_pos);
    const vec3 light_dir_reflect = reflect(-light_dir, normal);
    const float specular_intensity = pow(max(dot(view_dir, light_dir_reflect), 0.0), material.shininess);
    const vec3 specular_result = light.specular * specular_intensity * specular_color;

    const float f_dist = length(light.position - frag_pos);
    const float f_dist2 = f_dist * f_dist;
    const float attenuation =
        1.0 / (
            light.attenuation.constant +
            f_dist * light.attenuation.linear +
            f_dist2 * light.attenuation.quadratic);

    return attenuation * (ambient_result + diffuse_result + specular_result);
}

void main() {
    const float ambient_factor = 0.1;
    const vec3 diffuse = texture(material.diffuse, uv).rgb;
    const vec3 specular = texture(material.specular, uv).rgb;

    vec3 color = diffuse * ambient_factor;
    for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
        color += calculate_point_light(point_lights[i], diffuse, specular);
    }

    pixel = vec4(color, 1.0);
}
