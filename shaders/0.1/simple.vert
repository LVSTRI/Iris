#version 330 core

layout (location = 0) in vec3 position;

void main() {
    // gl_Position is a global predefined variable that serves as the *output* of the vertex shader.
    // the w component is used internally by OpenGL to perform *perspective division*.
    gl_Position = vec4(position.x, position.y, position.z, 1.0);
}
