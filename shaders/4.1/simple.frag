#version 460 core
#define CASCADE_COUNT 4

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

struct cascade_data_t {
    mat4 projection;
    mat4 view;
    mat4 pv;
    mat4 global;
    vec4 scale;
    vec4 offset; // w is split
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
layout (location = 8) uniform sampler2DArrayShadow shadow_map;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (std430, binding = 2) readonly restrict buffer cascade_output_t {
    cascade_data_t[CASCADE_COUNT] cascades;
};

layout (std140, binding = 3) readonly restrict buffer point_light_uniform_t {
    point_light_t[] data;
} point_light;

layout (std140, binding = 4) uniform directional_light_uniform_t {
    directional_light_t dir_light;
};

uint calculate_cascade() {
    const float dist = abs((camera.projection * camera.view * vec4(frag_pos, 1.0)).w);
    for (uint i = 0; i < num_cascades; ++i) {
        if (dist < cascades[i].offset.w) {
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

vec2 calcualte_depth_plane_bias(in vec3 ddx, in vec3 ddy) {
    vec2 bias_uv = vec2(
        ddy.y * ddx.z - ddx.y * ddy.z,
        ddx.x * ddy.z - ddy.x * ddx.z);
    bias_uv *= 1.0 / ((ddx.x * ddy.y) - (ddx.y * ddy.x));
    return bias_uv;
}

vec3 sample_shadow(in vec3 shadow_frag_pos, in vec3 ddx_shadow_frag_pos, in vec3 ddy_shadow_frag_pos, in vec3 normal, in uint cascade) {
    shadow_frag_pos += cascades[cascade].offset.xyz;
    shadow_frag_pos *= cascades[cascade].scale.xyz;
    ddx_shadow_frag_pos *= cascades[cascade].scale.xyz;
    ddy_shadow_frag_pos *= cascades[cascade].scale.xyz;


    const vec2 shadow_size = vec2(textureSize(shadow_map, 0));
    const vec2 texel_size = 1.0 / shadow_size;

    const float plane_bias = min(dot(vec2(1.0) * texel_size, abs(calcualte_depth_plane_bias(ddx_shadow_frag_pos, ddy_shadow_frag_pos))), 0.01);

    const vec3 n_light_dir = normalize(dir_light.direction);
    const float n_dot_l = dot(normal, n_light_dir);
    const float width = plane_bias * (1 / (8.0 * float(cascade + 1)));
    const float bias = clamp((width / 2.0) * tan(acos(clamp(n_dot_l, -1.0, 1.0))), 0.0, width);
    const float light_depth = shadow_frag_pos.z - bias;
    const vec2 sampling_texel_size = 1.0 / shadow_size;
    vec3 shadow_factor = vec3(0.0);
    uint sampled_count = 0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            const vec2 offset = vec2(x, y);
            const vec2 s_uv = shadow_frag_pos.xy + (offset * sampling_texel_size);
            shadow_factor += texture(shadow_map, vec4(s_uv, cascade, light_depth)).r;
            ++sampled_count;
        }
    }
    return shadow_factor / sampled_count;
}

vec3 calculate_shadow(in vec3 normal, in uint cascade) {
    const vec3 shadow_frag_pos = vec3(cascades[cascade].global * vec4(frag_pos, 1.0));
    const vec3 ddx_shadow_frag_pos = dFdxFine(shadow_frag_pos);
    const vec3 ddy_shadow_frag_pos = dFdyFine(shadow_frag_pos);
    return sample_shadow(shadow_frag_pos, ddx_shadow_frag_pos, ddy_shadow_frag_pos, normal, cascade);
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

    /*switch (cascade) {
        case 0: color *= vec3(1.0, 0.5, 0.5); break;
        case 1: color *= vec3(0.5, 1.0, 0.5); break;
        case 2: color *= vec3(0.5, 0.5, 1.0); break;
        case 3: color *= vec3(1.0, 1.0, 0.5); break;
    }*/
    out_pixel = vec4(color, alpha);
    out_transform_id = transform_id;
}
