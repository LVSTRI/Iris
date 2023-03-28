#version 460 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec3 out_frag_pos;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec2 out_uv;

layout (location = 0) uniform mat4 projection;
layout (location = 1) uniform mat4 view;
layout (location = 2) uniform mat4 model;
layout (location = 3) uniform mat4 t_inv_model;

void main() {
    const vec4 frag_pos = model * vec4(position, 1.0);
    out_frag_pos = vec3(frag_pos);
    out_normal = normalize(mat3(t_inv_model) * normal);
    out_uv = uv;

    gl_Position = projection * view * frag_pos;
}
