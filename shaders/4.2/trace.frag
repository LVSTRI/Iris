#version 460 core

layout (location = 0) out vec4 out_pixel;

/*struct camera_data_t {
    float aspect;
    float focal_length;
    float viewport_height;
    float viewport_width;
    vec4 origin;
    vec4 horizontal;
    vec4 vertical;
    vec4 lower_left_corner;
};*/

struct camera_data_t {
    mat4 inv_pv;
};

layout (std140, binding = 0) uniform camera_buffer_t {
    camera_data_t camera;
};

struct ray_t {
    vec3 origin;
    vec3 direction;
};

vec3 ray_hit(in ray_t ray, in float t) {
    return ray.origin + t * ray.direction;
}

const uint E_HITTABLE_NONE = 0;
const uint E_HITTABLE_SPHERE = 1;
const uint E_HITTABLE_TRIANGLE = 2;

struct hittable_t {
    uint type;
};

struct sphere_t {
    hittable_t hittable;
    vec3 center;
    float radius;
    uint material_id;
};

struct _proxy_hittable_t {
    // should be the max size of all hittable types or more
    uint[8] _data;
};

struct hit_record_t {
    float t;
    vec3 point;
    vec3 normal;
    bool front_face;
    uint material_id;
};

layout (std430, binding = 1) readonly restrict buffer hittable_buffer_t {
    _proxy_hittable_t[] hittables;
};

const uint E_MATERIAL_NONE = 0;
const uint E_MATERIAL_LAMBERTIAN = 1;
const uint E_MATERIAL_METAL = 2;
const uint E_MATERIAL_DIELECTRIC = 3;

struct material_t {
    uint type;
};

struct lambertian_t {
    material_t material;
    vec3 albedo;
    vec3 emissive;
    float e_strength;
};

struct metal_t {
    material_t material;
    vec3 albedo;
    float fuzz;
};

struct dielectric_t {
    material_t material;
    float refr_index;
};

struct _proxy_material_t {
    // should be the max size of all material types or more
    uint[8] _data;
};

layout (std430, binding = 2) readonly restrict buffer material_buffer_t {
    _proxy_material_t[] materials;
};

layout (location = 0) uniform vec2 resolution;
layout (location = 1) uniform uint frame;
layout (location = 2) uniform float time;

bool near_zero(in vec3 v) {
    return all(lessThan(abs(v), vec3(1e-8)));
}

uint state_init_prng() {
    return uint(time + 1) * 719393u + uint(gl_FragCoord.x + gl_FragCoord.y * resolution.x);
}

uint wang_hash(in uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return 1u + seed;
}

uint xorshift32(uint state) {
    uint x = state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

uint pcg(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(inout uint state) {
    state = pcg(state);
    return float(state) / float(uint(-1));
}

float random(inout uint state, in float min, in float max) {
    return min + (max - min) * random(state);
}

vec2 random_vec2(inout uint state) {
    return vec2(random(state), random(state));
}

vec2 random_vec2(inout uint state, in float min, in float max) {
    return min + (max - min) * random_vec2(state);
}

vec3 random_vec3(inout uint state) {
    return vec3(random(state), random(state), random(state));
}

vec3 random_vec3(inout uint state, in float min, in float max) {
    return min + (max - min) * random_vec3(state);
}

vec4 random_vec4(inout uint state) {
    return vec4(random(state), random(state), random(state), random(state));
}

vec4 random_vec4(inout uint state, in float min, in float max) {
    return min + (max - min) * random_vec4(state);
}

vec3 random_in_unit_sphere(inout uint state) {
    for (uint i = 0; i < 16; ++i) {
        const vec3 p = random_vec3(state, -1.0, 1.0);
        if (dot(p, p) >= 1.0) {
            continue;
        }
        return p;
    }
    return random_vec3(state, -1.0, 1.0) / 4.0;
}

vec3 random_in_hemisphere(inout uint state, in vec3 normal) {
    const vec3 in_unit_sphere = random_in_unit_sphere(state);
    if (dot(in_unit_sphere, normal) > 0.0) {
        return in_unit_sphere;
    }
    return -in_unit_sphere;
}

uint as_type_from_hittable_proxy(in _proxy_hittable_t proxy) {
    return proxy._data[0];
}

sphere_t as_sphere_from_hittable_proxy(in _proxy_hittable_t proxy) {
    sphere_t sphere;
    sphere.hittable.type = proxy._data[0];
    sphere.center = vec3(
        uintBitsToFloat(proxy._data[1]),
        uintBitsToFloat(proxy._data[2]),
        uintBitsToFloat(proxy._data[3]));
    sphere.radius = uintBitsToFloat(proxy._data[4]);
    sphere.material_id = proxy._data[5];
    return sphere;
}

uint as_type_from_material_proxy(in _proxy_material_t proxy) {
    return proxy._data[0];
}

lambertian_t as_lambertian_from_material_proxy(in _proxy_material_t proxy) {
    lambertian_t lambertian;
    lambertian.material.type = proxy._data[0];
    lambertian.albedo = vec3(
        uintBitsToFloat(proxy._data[1]),
        uintBitsToFloat(proxy._data[2]),
        uintBitsToFloat(proxy._data[3]));
    lambertian.emissive = vec3(
        uintBitsToFloat(proxy._data[4]),
        uintBitsToFloat(proxy._data[5]),
        uintBitsToFloat(proxy._data[6]));
    lambertian.e_strength = uintBitsToFloat(proxy._data[7]);
    return lambertian;
}

metal_t as_metal_from_material_proxy(in _proxy_material_t proxy) {
    metal_t metal;
    metal.material.type = proxy._data[0];
    metal.albedo = vec3(
        uintBitsToFloat(proxy._data[1]),
        uintBitsToFloat(proxy._data[2]),
        uintBitsToFloat(proxy._data[3]));
    metal.fuzz = uintBitsToFloat(proxy._data[4]);
    return metal;
}

dielectric_t as_dielectric_from_material_proxy(in _proxy_material_t proxy) {
    dielectric_t dielectric;
    dielectric.material.type = proxy._data[0];
    dielectric.refr_index = uintBitsToFloat(proxy._data[1]);
    return dielectric;
}

float reflectance(in float cosine, in float refr_index) {
    const float r0 = (1.0 - refr_index) / (1.0 + refr_index);
    const float r0_2 = r0 * r0;
    return r0_2 + (1.0 - r0_2) * pow(1.0 - cosine, 5.0);
}

bool scatter_lambertian(in lambertian_t material,
                        in ray_t ray,
                        in hit_record_t record,
                        out vec3 attenuation,
                        out vec3 emitted,
                        out ray_t scattered,
                        inout uint state) {
    vec3 scatter_dir = record.normal + normalize(random_in_unit_sphere(state));
    if (near_zero(scatter_dir)) {
        scatter_dir = record.normal;
    }
    scattered = ray_t(record.point, scatter_dir);
    attenuation = material.albedo;
    emitted = material.emissive * material.e_strength;
    return true;
}

bool scatter_metal(in metal_t material,
                   in ray_t ray,
                   in hit_record_t record,
                   out vec3 attenuation,
                   out ray_t scattered,
                   inout uint state) {
    const vec3 reflected = reflect(normalize(ray.direction), record.normal);
    scattered = ray_t(record.point, reflected);
    attenuation = material.albedo;
    return dot(scattered.direction, record.normal) > 0.0;
}

bool scatter_dielectric(in dielectric_t material,
                        in ray_t ray,
                        in hit_record_t record,
                        out vec3 attenuation,
                        out ray_t scattered,
                        inout uint state) {
    attenuation = vec3(1.0);
    const float refr_ratio = record.front_face ? (1.0 / material.refr_index) : material.refr_index;
    const vec3 r_dir = normalize(ray.direction);
    const float cos_t = min(dot(-r_dir, record.normal), 1.0);
    const float sin_t = sqrt(1.0 - cos_t * cos_t);
    const bool cannot_refract = refr_ratio * sin_t > 1.0;
    vec3 direction;
    if (cannot_refract || reflectance(cos_t, refr_ratio) > random(state)) {
        direction = reflect(r_dir, record.normal);
    } else {
        direction = refract(r_dir, record.normal, refr_ratio);
    }
    scattered = ray_t(record.point, direction);
    return true;
}

bool global_scatter(in _proxy_material_t proxy,
                    in ray_t ray,
                    in hit_record_t record,
                    out vec3 attenuation,
                    out vec3 emitted,
                    out ray_t scattered,
                    inout uint state) {
    switch (as_type_from_material_proxy(proxy)) {
        case E_MATERIAL_NONE:
            return false;

        case E_MATERIAL_LAMBERTIAN:
            return scatter_lambertian(as_lambertian_from_material_proxy(proxy), ray, record, attenuation, emitted, scattered, state);

        case E_MATERIAL_METAL:
            return scatter_metal(as_metal_from_material_proxy(proxy), ray, record, attenuation, scattered, state);

        case E_MATERIAL_DIELECTRIC:
            return scatter_dielectric(as_dielectric_from_material_proxy(proxy), ray, record, attenuation, scattered, state);
    }
    emitted = vec3(0);
    return false;
}

bool hit_sphere(in sphere_t sphere, in ray_t ray, in float t_min, in float t_max, out hit_record_t record) {
    const vec3 oc = ray.origin - sphere.center;
    const float a = dot(ray.direction, ray.direction);
    const float b = dot(oc, ray.direction);
    const float c = dot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = b * b - a * c;
    if (discriminant < 0.0) {
        return false;
    }
    float sq_d = sqrt(discriminant);
    float root = (-b - sq_d) / a;
    if (root < t_min || t_max < root) {
        root = (-b + sq_d) / a;
        if (root < t_min || t_max < root) {
            return false;
        }
    }
    record.t = root;
    record.point = ray_hit(ray, record.t);
    const vec3 normal = (record.point - sphere.center) / sphere.radius;
    const bool front_face = dot(ray.direction, normal) < 0.0;
    record.normal = front_face ? normal : -normal;
    record.front_face = front_face;
    record.material_id = sphere.material_id;
    return true;
}

bool global_hit(in _proxy_hittable_t proxy, in ray_t ray, in float t_min, in float t_max, out hit_record_t record) {
    const uint hittable_type = proxy._data[0];
    switch (hittable_type) {
        case E_HITTABLE_NONE:
            return false;

        case E_HITTABLE_SPHERE:
            return hit_sphere(as_sphere_from_hittable_proxy(proxy), ray, t_min, t_max, record);

        case E_HITTABLE_TRIANGLE:
            return false;
    }
    return false;
}

bool world_hit(in ray_t ray, in float t_min, in float t_max, out hit_record_t record) {
    bool hit_anything = false;
    float closest = t_max;
    for (uint i = 0; as_type_from_hittable_proxy(hittables[i]) != E_HITTABLE_NONE; ++i) {
        hit_record_t current_hit;
        if (global_hit(hittables[i], ray, t_min, closest, current_hit)) {
            hit_anything = true;
            closest = current_hit.t;
            record = current_hit;
        }
    }
    return hit_anything;
}

vec3 ray_color(in ray_t ray, inout uint state) {
    hit_record_t record;
    vec3 incoming_light = vec3(0);
    vec3 color = vec3(1.0);
    uint hits = 0;
    ray_t t_ray = ray;
    while (world_hit(t_ray, 0.01, 100000.0, record)) {
        if (hits++ == 32) {
            return vec3(0.0);
        }
        ray_t scattered;
        vec3 attenuation;
        vec3 emitted;
        if (global_scatter(materials[record.material_id], t_ray, record, attenuation, emitted, scattered, state)) {
            incoming_light += emitted * color;
            color *= attenuation;
            t_ray = scattered;
        } else {
            return vec3(0.0);
        }
    }
    const float intensity = 0.9325;
    const vec3 n_dir = normalize(t_ray.direction);
    const float t = 0.5 * (n_dir.y + 1.0);
    const vec3 sky_gradient = vec3(0.4, 0.4, 1.0);
    return (mix(vec3(1.0), sky_gradient, t) * color * intensity) + incoming_light;
}

void main() {
    const uint spp = 4;
    uint rng_state = state_init_prng();
    vec3 color = vec3(0.0);
    for (uint i = 0; i < spp; ++i) {
        const vec2 uv = (vec2(gl_FragCoord.xy) + random_vec2(rng_state, -1.0, 1.0)) / (vec2(resolution) - 1);
        const vec2 ndc_uv = 2.0 * uv - 1.0;
        const vec4 ndc_uv_near = vec4(ndc_uv, -1.0, 1.0);
        const vec4 ndc_uv_far = vec4(ndc_uv, 1.0, 1.0);
        vec4 world_near = camera.inv_pv * ndc_uv_near;
        vec4 world_far = camera.inv_pv * ndc_uv_far;
        world_near /= world_near.w;
        world_far /= world_far.w;

        ray_t ray;
        ray.origin = vec3(world_near);
        ray.direction = normalize(vec3(world_far - world_near));

        color += ray_color(ray, rng_state);
    }
    out_pixel = vec4(color / float(spp), 1.0);
}
