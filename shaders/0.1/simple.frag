#version 330 core

out vec4 pixel;

void main() {
    // every fragment will have the same color.
    pixel = vec4(1.0, 0.5, 0.2, 1.0);
}
