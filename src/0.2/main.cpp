#include <array>

#include <texture.hpp>
#include <shader.hpp>
#include <utilities.hpp>

#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <stb_image.h>

using namespace iris::literals;

static auto process_keyboard_input(GLFWwindow* window) noexcept -> void {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        // notifies the window that it should close
        glfwSetWindowShouldClose(window, true);
    }
}

constexpr auto WINDOW_WIDTH = 800;
constexpr auto WINDOW_HEIGHT = 600;

int main() {
    // GLFW initialization
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // window creation
    auto* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Hello World", nullptr, nullptr);
    if (!window) {
        iris::log("err: failed to create GLFW window");
        glfwTerminate();
        return -1;
    }

    // tie the OpenGL context to the GLFW window (also ties the window and the context to the current thread)
    glfwMakeContextCurrent(window);

    // use GLAD to load OpenGL function pointers (using GLFW's function loader)
    if (!gladLoadGL(glfwGetProcAddress)) {
        iris::log("err: failed to initialize GLAD");
        glfwTerminate();
        return -1;
    }

    // sets the viewport to the entire window
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // set callback on window resize: update the viewport.
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
        iris::log("window resize: ", width, "x", height);
        glViewport(0, 0, width, height);
    });

    // stbi should flip y-axis for OpenGL
    stbi_set_flip_vertically_on_load(true);

    // vertex and fragment shader creation
    auto shader = iris::shader_t::create("../shaders/0.2/simple.vert", "../shaders/0.2/simple.frag");

    // texture loading
    auto texture = iris::texture_t::create("../textures/wall.jpg");

    constexpr auto quad_vertices = std::to_array({
        -0.75f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, // top left
        -0.75f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
        -0.25f, -0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
        -0.25f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, // top right
    });
    constexpr auto triangle_vertices = std::to_array({
        0.25f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.75f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        0.5f,   0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.5f, 1.0f,
    });
    constexpr auto quad_indices = std::to_array<iris::uint32>({
        0, 1, 2,
        0, 2, 3
    });
    constexpr auto triangle_indices = std::to_array<iris::uint32>({
        0, 1, 2
    });

    // vertex array object (VAO) creation
    auto vaos = std::vector<iris::uint32>(2);
    auto vbos = std::vector<iris::uint32>(2);
    auto ebos = std::vector<iris::uint32>(2);

    glGenVertexArrays(vaos.size(), vaos.data());
    glGenBuffers(vbos.size(), vbos.data());
    glGenBuffers(ebos.size(), ebos.data());

    // VBO initialization
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(quad_vertices), quad_vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(triangle_vertices), triangle_vertices.data(), GL_STATIC_DRAW);

    // EBO initialization
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, iris::size_bytes(quad_indices), quad_indices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, iris::size_bytes(triangle_indices), triangle_indices.data(), GL_STATIC_DRAW);

    // VAO initialization
    glBindVertexArray(vaos[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[0]);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)sizeof(iris::float32[3]));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)sizeof(iris::float32[6]));
    glEnableVertexAttribArray(2);


    glBindVertexArray(vaos[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[1]);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)sizeof(iris::float32[3]));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)sizeof(iris::float32[6]));
    glEnableVertexAttribArray(2);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        const auto current_time = glfwGetTime();

        glfwPollEvents();

        process_keyboard_input(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader
            .bind()
            .set(0, { 0 });

        texture.bind(0);

        glBindVertexArray(vaos[0]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        glBindVertexArray(vaos[1]);
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, nullptr);

        glfwSwapBuffers(window);
    }

    // cleanup.
    glDeleteVertexArrays(vaos.size(), vaos.data());
    glDeleteBuffers(vbos.size(), vbos.data());
    glDeleteBuffers(ebos.size(), ebos.data());
    glfwTerminate();
    return 0;
}
