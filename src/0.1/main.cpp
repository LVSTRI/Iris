#include <array>

#include "utilities.hpp"

#include "glad/gl.h"

#include "GLFW/glfw3.h"

using namespace iris::literals;

static auto process_keyboard_input(GLFWwindow* window) noexcept -> void {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        // notifies the window that it should close
        glfwSetWindowShouldClose(window, true);
    }
}

static auto shader_compile_status(iris::uint32 shader) -> void {
    auto success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        auto info = std::array<char, 1024>();
        glGetShaderInfoLog(shader, info.size(), nullptr, info.data());
        iris::log("err: shader compilation failed with: ", info.data());
    }
}

static auto program_link_status(iris::uint32 program) -> void {
    auto success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        auto info = std::array<char, 1024>();
        glGetProgramInfoLog(program, info.size(), nullptr, info.data());
        iris::log("err: shader program linking failed with: ", info.data());
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
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        iris::log("window resize: ", width, "x", height);
        glViewport(0, 0, width, height);
    });

    /*
     * graphics pipeline
     * primitive assembly                        | a logical stage that "assembles" the vertices into primitives (e.g. triangles) whenever the GPU deems necessary,
     *                                             usually in between vertex and geometry / geometry and rasterization.
     *                                             N.B.: since the geometry shader acts on primitives, primitive assembly *has* to happen before the geometry shader (assuming it is used).
     * vertex data         -> vertex shader      | transforms the input vertices from a 3D space to another 3D space (whatever it means).
     * vertex shader       -> geometry shader    | the output of the vertex shader is optionally processed by a geometry shader.
     * geometry shader     -> rasterization      | the primitives are transformed from 3D space to 2D space (i.e. the screen), primitives outside the screen are "clipped".
     * rasterization       -> fragment shader    | the fragment shader is called for each fragment generated from the non-clipped primitives, a fragment is a "pixel-sized piece" of a primitive, and it comes with associated
     *                                             data such as: pixel location, barycentric coordinates, etc. this stage is usually used to determine the "color" of a fragment (applying lighting effects, shadows, etc.).
     * N.B.: avoid geometry shaders like the plague.
     * both before and after the fragment shader stage more tests are performed on the final
     * result (e.g. depth test, stencil test, alpha test, etc.) and the final color is then determined.
     */

    // vertex and fragment shader creation
    auto shader_program = glCreateProgram();
    {
        auto vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        const auto vertex_shader_file = iris::whole_file("../shaders/simple.vert");
        const auto* vertex_shader_source = vertex_shader_file.c_str();
        glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
        glCompileShader(vertex_shader);
        shader_compile_status(vertex_shader);

        auto fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        const auto fragment_shader_file = iris::whole_file("../shaders/simple.frag");
        const auto* fragment_shader_source = fragment_shader_file.c_str();
        glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
        glCompileShader(fragment_shader);
        shader_compile_status(fragment_shader);

        glAttachShader(shader_program, vertex_shader);
        glAttachShader(shader_program, fragment_shader);
        glLinkProgram(shader_program);
        program_link_status(shader_program);

        glDeleteShader(fragment_shader);
        glDeleteShader(vertex_shader);
    }

    // normalized device coordinates: (x, y, z) range in [-1, 1] as per the OpenGL specification, they will be transformed to
    // screen-space coordinates with a *viewport transform* that uses the information provided in the "glViewport" call.
    constexpr auto triangle_vertices = std::to_array({
         0.5f, -0.5f, 0.0f, // bottom right vertex
        -0.5f, -0.5f, 0.0f, // bottom left vertex
         0.0f,  0.5f, 0.0f  // top vertex
    });

    // vertices are stored in a vertex buffer object (VBO), this buffer lives on the GPU for fast access.
    // it is *immutable* because updating data from the CPU to the GPU is very slow.
    auto vbo = 0_u32;
    // generate a buffer object.
    glGenBuffers(1, &vbo);
    // associate a buffer object with the buffer type used for VBOs (GL_ARRAY_BUFFER).
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // allocate memory and copy the vertices to the GPU, also marks the memory as immutable with GL_STATIC_DRAW.
    glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(triangle_vertices), triangle_vertices.data(), GL_STATIC_DRAW);

    // vertex array object (VAO) creation
    auto vao = 0_u32;
    // generate a vertex array object.
    glGenVertexArrays(1, &vao);
    // bind the vertex array object.
    glBindVertexArray(vao);
    // associate a vertex attrbute (e.g. position) with a vertex buffer object.
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // describe how OpenGL should interpret the data, in this case we have 1 attribute with 3 floats tightly packed in the VBO.
    glVertexAttribPointer(
        /* location   = */ 0,
        /* components = */ 3,
        /* type       = */ GL_FLOAT,
        /* normalized = */ GL_FALSE,
        /* stride     = */ sizeof(iris::float32[3]),
        /* offset     = */ (void*)0);
    // vertex attributes are disabled by default, enable them.
    glEnableVertexAttribArray(0);

    // render loop
    while (!glfwWindowShouldClose(window)) {
        // check for events (window closed, key pressed, etc.) also calls the callbacks (if necessary).
        glfwPollEvents();

        // handle keyboard inputs.
        process_keyboard_input(window);

        // set the color to clear the screen with.
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        // clear the screen with the previously set color.
        glClear(GL_COLOR_BUFFER_BIT);

        // bind the shader.
        glUseProgram(shader_program);
        // bind the VAO.
        glBindVertexArray(vao);
        // draw the triangle.
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // swap front and back buffer.
        glfwSwapBuffers(window);
    }

    // cleanup.
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shader_program);
    glfwTerminate();
    return 0;
}
