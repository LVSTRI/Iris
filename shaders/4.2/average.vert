#version 460 core

layout (location = 0) out vec2 out_uv;

void main() {
    // fullscreen triangle
    vec2 position[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0));

    gl_Position = vec4(position[gl_VertexID], 0.0, 1.0);
    out_uv = position[gl_VertexID] * 0.5 + 0.5;
}
