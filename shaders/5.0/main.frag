#version 460 core
#extension GL_ARB_bindless_texture : enable

#define CASCADE_COUNT 4
#define M_PI 3.1415926535897932384626433832795
#define M_GOLDEN 1.6180339887498948482045868343656
#define M_GOLDEN_CONJ 0.6180339887498948482045868343656

struct cascade_data_t {
    mat4 projection;
    mat4 view;
    mat4 pv;
    mat4 global;
    vec4 scale;
    vec4 offset; // w is split
};

struct directional_light_t {
    vec3 direction;
    vec3 diffuse;
    vec3 specular;
};

layout (early_fragment_tests) in;

layout (location = 0) in flat uint i_diffuse_texture;
layout (location = 1) in flat uint i_normal_texture;
layout (location = 2) in flat uint i_specular_texture;
layout (location = 3) in flat uint i_draw_id;
layout (location = 4) in vec3 i_normal;
layout (location = 5) in vec2 i_uv;
layout (location = 6) in vec3 i_frag_pos;
layout (location = 7) in mat3 i_TBN;

layout (location = 0) out vec4 o_pixel;

layout (location = 1) uniform sampler2DArrayShadow shadow_map;
layout (location = 2) uniform sampler2D blue_noise;

layout (std140, binding = 0) uniform u_camera {
    mat4 projection;
    mat4 view;
    mat4 pv;
    vec3 position;
    float near;
    float far;
} camera;

layout (std140, binding = 5) uniform b_directional_lights {
    directional_light_t[16] directional_lights;
};

layout (std430, binding = 6) readonly restrict buffer b_texture {
    sampler2D[] textures;
};

layout (std430, binding = 7) readonly restrict buffer b_cascade_output {
    cascade_data_t[CASCADE_COUNT] cascades;
};

// from https://www.shadertoy.com/view/7ssfWN
vec2 sample_hammersley(in uint i, in uint n) {
    const float s = 2.3283064365386963e-10;
    return vec2(
        float(i) / float(n),
        float(bitfieldReverse(i)) * s);
}

vec2 calcualte_depth_plane_bias(in vec3 ddx, in vec3 ddy) {
    vec2 bias_uv = vec2(
    ddy.y * ddx.z - ddx.y * ddy.z,
    ddx.x * ddy.z - ddy.x * ddx.z);
    bias_uv *= 1.0 / ((ddx.x * ddy.y) - (ddx.y * ddy.x));
    return bias_uv;
}

uint calculate_cascade(in float depth_vs) {
    for (uint i = 0; i < CASCADE_COUNT; ++i) {
        if (depth_vs < cascades[i].offset.w) {
            return i;
        }
    }
    return CASCADE_COUNT - 1;
}

vec3 calculate_directional_light(in directional_light_t light, in vec3 diffuse_color, in vec3 specular_color, in vec3 normal) {
    const vec3 light_dir = normalize(light.direction);
    const float diffuse_intensity = max(dot(light_dir, normal), 0.0);
    const vec3 diffuse_result = light.diffuse * diffuse_intensity * diffuse_color;

    const vec3 view_dir = normalize(camera.position - i_frag_pos);
    const vec3 halfway = normalize(light_dir + view_dir);
    // TODO
    const float specular_intensity = pow(max(dot(normal, halfway), 0.0), 32);
    const vec3 specular_result = light.specular * specular_intensity * specular_color;

    return diffuse_result + specular_result;
}

vec3 sample_shadow(in vec3 shadow_frag_pos,
                   in vec3 ddx_shadow_frag_pos,
                   in vec3 ddy_shadow_frag_pos,
                   in vec3 light_dir,
                   in vec3 normal,
                   in float depth_vs,
                   in uint cascade) {
    shadow_frag_pos += cascades[cascade].offset.xyz;
    shadow_frag_pos *= cascades[cascade].scale.xyz;
    ddx_shadow_frag_pos *= cascades[cascade].scale.xyz;
    ddy_shadow_frag_pos *= cascades[cascade].scale.xyz;

    const vec2 shadow_size = vec2(textureSize(shadow_map, 0));
    const vec2 texel_size = 1.0 / shadow_size;

    //const vec2 bias_uv = calcualte_depth_plane_bias(ddx_shadow_frag_pos, ddy_shadow_frag_pos);
    const float width = 0.0085;
    const vec3 halfway = normalize(light_dir + normal);
    float bias = max(
        clamp((width / 2.0) * tan(acos(abs(clamp(dot(normal, halfway), -1.0, 1.0)))), 0.0, width),
        clamp((width / 2.0) * tan(acos(abs(clamp(dot(halfway, light_dir), -1.0, 1.0)))), 0.0, width));
    //float bias = clamp((width / 2.0) * tan(acos(abs(clamp(dot(normal, light_dir), -1.0, 1.0)))), 0.0, width);
    bias /= float(cascade + 1);
    const float light_depth = shadow_frag_pos.z - bias;
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

vec3 calculate_shadow(in directional_light_t light, in vec3 normal, in float depth_vs, in uint cascade) {
    const vec3 shadow_frag_pos = vec3(cascades[cascade].global * vec4(i_frag_pos, 1.0));
    const vec3 ddx_shadow_frag_pos = dFdxFine(shadow_frag_pos);
    const vec3 ddy_shadow_frag_pos = dFdyFine(shadow_frag_pos);
    vec3 shadow_factor = sample_shadow(
        shadow_frag_pos,
        ddx_shadow_frag_pos,
        ddy_shadow_frag_pos,
        light.direction,
        normal,
        depth_vs,
        cascade);

    const float blend_threshold = 0.175;
    const float next_split = cascades[cascade].offset.w;
    const float split_size = cascade == 0 ? next_split : next_split - cascades[cascade - 1].offset.w;
    const float split_distance = (next_split - depth_vs) / split_size;
    if (split_distance <= blend_threshold && cascade != CASCADE_COUNT - 1) {
        const vec3 next_shadow_factor = sample_shadow(
            shadow_frag_pos,
            ddx_shadow_frag_pos,
            ddy_shadow_frag_pos,
            light.direction,
            normal,
            depth_vs,
            cascade + 1);
        const float t_lerp = smoothstep(0.0, blend_threshold, split_distance);
        shadow_factor = mix(next_shadow_factor, shadow_factor, t_lerp);
    }

    return shadow_factor;
}

vec3 hsv_to_rgb(in vec3 hsv) {
    const vec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsv.z * mix(vec3(1.0), rgb, hsv.y);
}

void main() {
    const float ambient_factor = 0.025;
    vec3 diffuse = vec3(1.0);
    if (i_diffuse_texture != -1) {
        diffuse = texture(textures[i_diffuse_texture], i_uv).rgb;
    }
    vec3 normal = normalize(i_normal);
    if (i_normal_texture != -1) {
        normal = normalize(i_TBN * (texture(textures[i_normal_texture], i_uv).rgb * 2.0 - 1.0));
    }
    vec3 specular = vec3(0.0);
    if (i_specular_texture != -1) {
        specular = texture(textures[i_specular_texture], i_uv).rgb;
    }
    const vec3 ambient = diffuse * ambient_factor;
    const float depth_vs = (camera.pv * vec4(i_frag_pos, 1.0)).w;
    const uint cascade = calculate_cascade(depth_vs);

    //vec3 hsv = vec3(fract(M_GOLDEN_CONJ * (i_draw_id * gl_PrimitiveID + 1)), 0.875, 0.85);
    //if (i_diffuse_texture == -1) {
    //    hsv = vec3(0.0, 0.0, 0.0);
    //}
    //diffuse = hsv_to_rgb(hsv);

    vec3 color = diffuse * ambient_factor;
    /*for (uint i = 0; i < n_point_lights; i++) {
        color += calculate_point_light(point_light.data[i], diffuse, specular, n_normal);
    }*/

    color +=
        calculate_directional_light(directional_lights[0], diffuse, specular, normal) *
        calculate_shadow(directional_lights[0], normal, depth_vs, cascade);

    o_pixel = vec4(color, 1.0);
}
