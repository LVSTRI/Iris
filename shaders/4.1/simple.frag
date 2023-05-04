#version 460 core
#define CASCADE_COUNT 4

#define M_PI 3.1415926535897932384626433832795

#define SHADOW_MODE_SOFT 0
#define SHADOW_MODE_HARD
#define SHADOW_MODE SHADOW_MODE_SOFT

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

layout (location = 4) uniform material_t material;
layout (location = 7) uniform uint n_point_lights;
layout (location = 8) uniform sampler2DArrayShadow shadow_map;
layout (location = 9) uniform sampler2D blue_noise;

layout (std140, binding = 0) uniform camera_uniform_t {
    mat4 projection;
    mat4 view;
    vec3 position;
    float near;
    float far;
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

// from https://www.shadertoy.com/view/7ssfWN
vec2 sample_hammersley(uint i, uint n) {
    const float s = 2.3283064365386963e-10;
    return vec2(
        float(i) / float(n),
        float(bitfieldReverse(i)) * s);
}

uint calculate_cascade(float depth_vs) {
    for (uint i = 0; i < CASCADE_COUNT; ++i) {
        if (depth_vs < cascades[i].offset.w) {
            return i;
        }
    }
    return CASCADE_COUNT - 1;
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

vec3 sample_shadow(in vec3 shadow_frag_pos, in vec3 ddx_shadow_frag_pos, in vec3 ddy_shadow_frag_pos, in vec3 normal, in float depth_vs, in uint cascade) {
    shadow_frag_pos += cascades[cascade].offset.xyz;
    shadow_frag_pos *= cascades[cascade].scale.xyz;
    ddx_shadow_frag_pos *= cascades[cascade].scale.xyz;
    ddy_shadow_frag_pos *= cascades[cascade].scale.xyz;

    const vec2 shadow_size = vec2(textureSize(shadow_map, 0));
    const vec2 texel_size = 1.0 / shadow_size;

    const vec2 bias_uv = calcualte_depth_plane_bias(ddx_shadow_frag_pos, ddy_shadow_frag_pos);
    const float plane_bias = min(dot(vec2(1.0) * texel_size, abs(bias_uv)), 0.005);
    const vec3 n_light_dir = normalize(dir_light.direction);
    const float n_dot_l = dot(normal, n_light_dir);
    float bias = clamp((plane_bias / 2.0) * tan(acos(clamp(n_dot_l, -1.0, 1.0))), 0.0, plane_bias);
    bias = bias * (1 / (8.0 * float(cascade + 1)));
    const float light_depth = shadow_frag_pos.z + bias;
    const float kernel_radius = float[](4.0, 3.0, 2.0, 1.0)[cascade];
    const uint sample_count = 64;
    vec3 shadow_factor = vec3(0.0);

    for (uint i = 0; i < sample_count; ++i) {
        const ivec2 noise_size = textureSize(blue_noise, 0);
        const ivec2 noise_texel = ivec2(int(gl_FragCoord.x) % noise_size.x, int(gl_FragCoord.y) % noise_size.y);
        const vec2 xi = fract(sample_hammersley(i, sample_count) + texelFetch(blue_noise, noise_texel, 0).xy);
        const float r = sqrt(xi.x);
        const float theta = xi.y * 2.0 * M_PI;
        const vec2 offset = vec2(r * cos(theta), r * sin(theta));
        const vec2 s_uv = shadow_frag_pos.xy + offset * texel_size * kernel_radius;
        shadow_factor += texture(shadow_map, vec4(s_uv, cascade, light_depth)).r;
    }
    vec3 shadow = shadow_factor / float(sample_count);
    /*switch (cascade) {
        case 0: shadow *= vec3(1.0, 0.25, 0.25); break;
        case 1: shadow *= vec3(0.25, 1.0, 0.25); break;
        case 2: shadow *= vec3(0.25, 0.25, 1.0); break;
        case 3: shadow *= vec3(1.0, 1.0, 0.25); break;
    }*/
    return shadow;
}

vec3 calculate_shadow(in vec3 normal, in float depth_vs, in uint cascade) {
    const vec3 shadow_frag_pos = vec3(cascades[cascade].global * vec4(frag_pos, 1.0));
    const vec3 ddx_shadow_frag_pos = dFdxFine(shadow_frag_pos);
    const vec3 ddy_shadow_frag_pos = dFdyFine(shadow_frag_pos);
    vec3 shadow_factor = sample_shadow(shadow_frag_pos, ddx_shadow_frag_pos, ddy_shadow_frag_pos, normal, depth_vs, cascade);

    const float blend_threshold = 0.175;
    const float next_split = cascades[cascade].offset.w;
    const float split_size = cascade == 0 ? next_split : next_split - cascades[cascade - 1].offset.w;
    const float split_distance = (next_split - depth_vs) / split_size;
    if (split_distance <= blend_threshold && cascade != CASCADE_COUNT - 1) {
        const vec3 next_shadow_factor = sample_shadow(shadow_frag_pos, ddx_shadow_frag_pos, ddy_shadow_frag_pos, normal, depth_vs, cascade + 1);
        const float t_lerp = smoothstep(0.0, blend_threshold, split_distance);
        shadow_factor = mix(next_shadow_factor, shadow_factor, t_lerp);
    }

    return shadow_factor;
}

void main() {
    const float ambient_factor = 0.01;
    const vec3 diffuse = texture(material.diffuse, uv).rgb;
    const float alpha = texture(material.diffuse, uv).a;
    const vec3 specular = texture(material.specular, uv).rgb;
    const vec3 ambient = diffuse * ambient_factor;
    const vec3 n_normal = normalize(normal);
    const float depth_vs = (camera.projection * camera.view * vec4(frag_pos, 1.0)).w;
    const uint cascade = calculate_cascade(depth_vs);

    vec3 color = ambient;
    for (uint i = 0; i < n_point_lights; i++) {
        color += calculate_point_light(point_light.data[i], diffuse, specular, n_normal);
    }

    color +=
        calculate_directional_light(diffuse, specular, n_normal) *
        calculate_shadow(n_normal, depth_vs, cascade);

    out_pixel = vec4(color, alpha);
    out_transform_id = transform_id;
}
