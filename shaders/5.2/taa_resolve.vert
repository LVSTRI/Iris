#version 460 core

layout (location = 0) out vec2 o_uv;

void main() {
    const vec2[] position = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0));
    o_uv = position[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(position[gl_VertexID], 0.0, 1.0);
}
