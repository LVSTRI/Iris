#include <numeric>
#include <cstdlib>
#include <vector>
#include <array>

#include <texture.hpp>
#include <shader.hpp>
#include <camera.hpp>
#include <utilities.hpp>

#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <stb_image.h>

using namespace iris::literals;

constexpr auto WINDOW_WIDTH = 800;
constexpr auto WINDOW_HEIGHT = 600;

struct shape_t {
    std::vector<iris::float32> vertices;
    std::vector<iris::uint32> indices;
    std::vector<glm::mat4> transforms;
};

static auto process_keyboard_input(GLFWwindow* window) noexcept -> void {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        // notifies the window that it should close
        glfwSetWindowShouldClose(window, true);
    }
}

static auto generate_cube() noexcept -> std::vector<iris::float32> {
    return std::vector<iris::float32> {
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f,
    };
}


int main() {
    srand(time(nullptr));
    // GLFW initialization
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // window creation
    auto window = iris::window_t();
    window.handle = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Hello World", nullptr, nullptr);
    window.width = WINDOW_WIDTH;
    window.height = WINDOW_HEIGHT;
    if (!window.handle) {
        iris::log("err: failed to create GLFW window");
        glfwTerminate();
        return -1;
    }
    glfwSetWindowUserPointer(window.handle, &window);

    // tie the OpenGL context to the GLFW window (also ties the window and the context to the current thread)
    glfwMakeContextCurrent(window.handle);

    // use GLAD to load OpenGL function pointers (using GLFW's function loader)
    if (!gladLoadGL(glfwGetProcAddress)) {
        iris::log("err: failed to initialize GLAD");
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback([] (GLenum source,
                               GLenum type,
                               GLuint id,
                               GLenum severity,
                               GLsizei length,
                               const GLchar* message,
                               const void*) {
        std::cout << "debug callback: " << message << std::endl;
    }, nullptr);

    // sets the viewport to the entire window
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // set callback on window resize: update the viewport.
    glfwSetFramebufferSizeCallback(window.handle, [](GLFWwindow* handle, int width, int height) {
        auto& window = *static_cast<iris::window_t*>(glfwGetWindowUserPointer(handle));
        iris::log("window resize: ", width, "x", height);
        glViewport(0, 0, width, height);
        window.width = width;
        window.height = height;
    });

    glfwSetMouseButtonCallback(window.handle, [](GLFWwindow* handle, int button, int action, int mods) {
        auto& window = *static_cast<iris::window_t*>(glfwGetWindowUserPointer(handle));
        if (glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            window.is_mouse_captured = true;
        } else if (glfwGetMouseButton(handle, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
            glfwSetInputMode(handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            window.is_mouse_captured = false;
        }
        window.cursor_position = {};
    });

    // stbi should flip y-axis for OpenGL
    stbi_set_flip_vertically_on_load(true);

    // camera initialization
    auto camera = iris::camera_t::create(window);

    // vertex and fragment shader creation
    auto simple_shader = iris::shader_t::create("../shaders/1.1/simple.vert", "../shaders/1.1/simple.frag");
    auto light_shader = iris::shader_t::create("../shaders/1.1/light.vert", "../shaders/1.1/light.frag");

    // texture loading
    auto wall = iris::texture_t::create("../textures/wall.jpg");
    auto container = iris::texture_t::create("../textures/container.png");
    auto container_specular = iris::texture_t::create("../textures/container_specular.png");

    auto cube = shape_t {
        generate_cube(),

        std::vector<iris::uint32>(36)
    };
    std::iota(cube.indices.begin(), cube.indices.end(), 0);

    auto shapes = std::vector<shape_t> {
        std::move(cube)
    };
    {
        auto& transforms = shapes[0].transforms;
        transforms = std::vector<glm::mat4>(10 + 4);
        for (auto i = 0_u32; i < 10; ++i) {
            auto angle = 20.0f * i;
            transforms[i] = glm::identity<glm::mat4>();
            transforms[i] = glm::rotate(transforms[i], glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            auto displacement = 2.5f * glm::vec3(
                glm::sin(glm::radians(angle * 2.0f)),
                glm::cos(glm::radians(angle * 2.0f)),
                glm::sin(glm::radians(angle * 2.0f)));
            transforms[i] = glm::translate(transforms[i], displacement);
        }
        for (auto i = 0_u32; i < 4; ++i) {
            transforms[i + 10] = glm::identity<glm::mat4>();
            transforms[i + 10] = glm::translate(transforms[i + 10], glm::vec3(
                2.0f - (4 * (rand() / (iris::float32)RAND_MAX)),
                2 * (rand() / (iris::float32)RAND_MAX),
                2.0f - (4 * (rand() / (iris::float32)RAND_MAX))));
            transforms[i + 10] = glm::scale(transforms[i + 10], glm::vec3(0.25f));
        }
    }

    // vertex array object (VAO) creation
    auto vaos = std::vector<iris::uint32>(shapes.size());
    auto vbos = std::vector<iris::uint32>(shapes.size());
    auto ebos = std::vector<iris::uint32>(shapes.size());

    glGenVertexArrays(vaos.size(), vaos.data());
    glGenBuffers(vbos.size(), vbos.data());
    glGenBuffers(ebos.size(), ebos.data());

    for (iris::uint32 i = 0; const auto& shape : shapes) {
        glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
        glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(shape.vertices), shape.vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[i]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, iris::size_bytes(shape.indices), shape.indices.data(), GL_STATIC_DRAW);
        i++;
    }

    // VAO initialization
    for (auto i = 0_u32; i < shapes.size(); ++i) {
        glBindVertexArray(vaos[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[i]);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)sizeof(iris::float32[3]));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(iris::float32[8]), (void*)sizeof(iris::float32[6]));
        glEnableVertexAttribArray(2);
    }

    // enable depth testing.
    glEnable(GL_DEPTH_TEST);

    // timing.
    auto delta_time = 0.0f;
    auto last_frame = 0.0f;

    // render loop
    while (!glfwWindowShouldClose(window.handle)) {
        const auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_frame;
        last_frame = current_time;

        process_keyboard_input(window.handle);
        // camera setup
        const auto projection = camera.projection();
        const auto view = camera.view();

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto shape = 0_u32;

        const auto light_pos = glm::vec3(2 * glm::sin(current_time), 1.0f * glm::sin(current_time), 2 * glm::cos(current_time));
        auto transform = glm::identity<glm::mat4>();
        transform = glm::translate(transform, light_pos);
        transform = glm::scale(transform, glm::vec3(0.2f));
        shapes[shape].transforms[10] = transform;

        for (auto i = 0_u32; i < 10; ++i) {
            container.bind(0);
            container_specular.bind(1);
            simple_shader
                .bind()
                .set(0, projection)
                .set(1, view)
                .set(2, shapes[shape].transforms[i])
                .set(3, glm::inverseTranspose(shapes[shape].transforms[i]))
                .set(4, camera.position())
                // material
                .set(5, { 0 })
                .set(6, { 1 })
                .set(7, { 32_u32 });
            for (auto j = 0_u32; j < 4; ++j) {
                simple_shader
                    // point lights
                    .set(8 + (7 * j), shapes[shape].transforms[10 + j])
                    .set(9 + (7 * j), { 0.1f, 0.1f, 0.1f })
                    .set(10 + (7 * j), { 0.5f, 0.5f, 0.5f })
                    .set(11 + (7 * j), { 1.0f, 1.0f, 1.0f })
                    .set(12 + (7 * j), { 1.0f })
                    .set(13 + (7 * j), { 0.09f })
                    .set(14 + (7 * j), { 0.032f });
            }
            simple_shader
                // directional light
                .set(36, glm::vec3(150.0f, 450.0f, 250.0f))
                .set(37, { 0.1f, 0.1f, 0.1f })
                .set(38, { 0.5f, 0.5f, 0.5f })
                .set(39, { 1.0f, 1.0f, 1.0f });

            glBindVertexArray(vaos[shape]);
            glDrawElements(GL_TRIANGLES, shapes[shape].indices.size(), GL_UNSIGNED_INT, nullptr);
        }

        for (auto i = 0_u32; i < 4; ++i) {
            shape = 0;
            light_shader
                .bind()
                .set(0, projection)
                .set(1, view)
                .set(2, shapes[shape].transforms[10 + i])
                .set(3, { 1.0f, 1.0f, 1.0f });

            glBindVertexArray(vaos[shape]);
            glDrawElements(GL_TRIANGLES, shapes[shape].indices.size(), GL_UNSIGNED_INT, nullptr);
        }

        glfwSwapBuffers(window.handle);
        glfwPollEvents();
        window.update();
        camera.update(delta_time);
    }

    // cleanup.
    glDeleteVertexArrays(vaos.size(), vaos.data());
    glDeleteBuffers(vbos.size(), vbos.data());
    glDeleteBuffers(ebos.size(), ebos.data());
    glfwTerminate();
    return 0;
}
