#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <array>
#include <functional>

#include <texture.hpp>
#include <shader.hpp>
#include <camera.hpp>
#include <utilities.hpp>
#include <mesh.hpp>
#include <model.hpp>
#include <framebuffer.hpp>

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

static auto generate_cube() noexcept {
    return std::vector<iris::vertex_t>({
        { { -0.5f, -0.5f, -0.5f }, {  0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f } },
        { {  0.5f, -0.5f, -0.5f }, {  0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f } },
        { {  0.5f,  0.5f, -0.5f }, {  0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f } },
        { {  0.5f,  0.5f, -0.5f }, {  0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f } },
        { { -0.5f,  0.5f, -0.5f }, {  0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f } },
        { { -0.5f, -0.5f, -0.5f }, {  0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f } },
        { { -0.5f, -0.5f,  0.5f }, {  0.0f, 0.0f,  1.0f }, { 0.0f, 0.0f } },
        { {  0.5f, -0.5f,  0.5f }, {  0.0f, 0.0f,  1.0f }, { 1.0f, 0.0f } },
        { {  0.5f,  0.5f,  0.5f }, {  0.0f, 0.0f,  1.0f }, { 1.0f, 1.0f } },
        { {  0.5f,  0.5f,  0.5f }, {  0.0f, 0.0f,  1.0f }, { 1.0f, 1.0f } },
        { { -0.5f,  0.5f,  0.5f }, {  0.0f, 0.0f,  1.0f }, { 0.0f, 1.0f } },
        { { -0.5f, -0.5f,  0.5f }, {  0.0f, 0.0f,  1.0f }, { 0.0f, 0.0f } },
        { { -0.5f,  0.5f,  0.5f }, { -1.0f, 0.0f,  0.0f }, { 1.0f, 0.0f } },
        { { -0.5f,  0.5f, -0.5f }, { -1.0f, 0.0f,  0.0f }, { 1.0f, 1.0f } },
        { { -0.5f, -0.5f, -0.5f }, { -1.0f, 0.0f,  0.0f }, { 0.0f, 1.0f } },
        { { -0.5f, -0.5f, -0.5f }, { -1.0f, 0.0f,  0.0f }, { 0.0f, 1.0f } },
        { { -0.5f, -0.5f,  0.5f }, { -1.0f, 0.0f,  0.0f }, { 0.0f, 0.0f } },
        { { -0.5f,  0.5f,  0.5f }, { -1.0f, 0.0f,  0.0f }, { 1.0f, 0.0f } },
        { {  0.5f,  0.5f,  0.5f }, {  1.0f, 0.0f,  0.0f }, { 1.0f, 0.0f } },
        { {  0.5f,  0.5f, -0.5f }, {  1.0f, 0.0f,  0.0f }, { 1.0f, 1.0f } },
        { {  0.5f, -0.5f, -0.5f }, {  1.0f, 0.0f,  0.0f }, { 0.0f, 1.0f } },
        { {  0.5f, -0.5f, -0.5f }, {  1.0f, 0.0f,  0.0f }, { 0.0f, 1.0f } },
        { {  0.5f, -0.5f,  0.5f }, {  1.0f, 0.0f,  0.0f }, { 0.0f, 0.0f } },
        { {  0.5f,  0.5f,  0.5f }, {  1.0f, 0.0f,  0.0f }, { 1.0f, 0.0f } },
        { { -0.5f, -0.5f, -0.5f }, {  0.0f, 1.0f,  0.0f }, { 0.0f, 1.0f } },
        { {  0.5f, -0.5f, -0.5f }, {  0.0f, 1.0f,  0.0f }, { 1.0f, 1.0f } },
        { {  0.5f, -0.5f,  0.5f }, {  0.0f, 1.0f,  0.0f }, { 1.0f, 0.0f } },
        { {  0.5f, -0.5f,  0.5f }, {  0.0f, 1.0f,  0.0f }, { 1.0f, 0.0f } },
        { { -0.5f, -0.5f,  0.5f }, {  0.0f, 1.0f,  0.0f }, { 0.0f, 0.0f } },
        { { -0.5f, -0.5f, -0.5f }, {  0.0f, 1.0f,  0.0f }, { 0.0f, 1.0f } },
        { { -0.5f,  0.5f, -0.5f }, {  0.0f, 1.0f,  0.0f }, { 0.0f, 1.0f } },
        { {  0.5f,  0.5f, -0.5f }, {  0.0f, 1.0f,  0.0f }, { 1.0f, 1.0f } },
        { {  0.5f,  0.5f,  0.5f }, {  0.0f, 1.0f,  0.0f }, { 1.0f, 0.0f } },
        { {  0.5f,  0.5f,  0.5f }, {  0.0f, 1.0f,  0.0f }, { 1.0f, 0.0f } },
        { { -0.5f,  0.5f,  0.5f }, {  0.0f, 1.0f,  0.0f }, { 0.0f, 0.0f } },
        { { -0.5f,  0.5f, -0.5f }, {  0.0f, 1.0f,  0.0f }, { 0.0f, 1.0f } },
    });
}

struct scene_t {
    std::vector<std::reference_wrapper<const iris::mesh_t>> opaque_meshes;
    std::vector<std::reference_wrapper<const iris::mesh_t>> transparent_meshes;
};

int main() {
    srand(time(nullptr));
    // GLFW initialization
    if (!glfwInit()) {
        iris::log("failed to initialize GLFW");
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto terminate_glfw = iris::defer_t([]() {
        glfwTerminate();
    });

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

#if !defined(NDEBUG)
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback([] (GLenum source,
                               GLenum type,
                               GLuint id,
                               GLenum severity,
                               GLsizei length,
                               const GLchar* message,
                               const void*) {
        if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
            return;
        }
        std::cout << "debug callback: " << message << std::endl;
        if (severity == GL_DEBUG_SEVERITY_HIGH) {
            std::terminate();
        }
    }, nullptr);
#endif

    // sets the viewport to the entire window
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    glfwFocusWindow(window.handle);

    // set callback on window resize: update the viewport.
    glfwSetFramebufferSizeCallback(window.handle, [](GLFWwindow* handle, int width, int height) {
        auto& window = *static_cast<iris::window_t*>(glfwGetWindowUserPointer(handle));
        iris::log("window resize: ", width, "x", height);
        glViewport(0, 0, width, height);
        window.width = width;
        window.height = height;
        window.is_resized = true;
    });

    glfwSetWindowFocusCallback(window.handle, [](GLFWwindow* handle, int focus) {
        auto& window = *static_cast<iris::window_t*>(glfwGetWindowUserPointer(handle));
        window.is_focused = focus;
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

    // camera initialization
    auto camera = iris::camera_t::create(window);

    // vertex and fragment shader creation
    auto screen_shader = iris::shader_t::create("../shaders/3.1/fullscreen.vert", "../shaders/3.1/fullscreen.frag");
    auto simple_shader = iris::shader_t::create("../shaders/3.1/simple.vert", "../shaders/3.1/simple.frag");
    auto light_shader = iris::shader_t::create("../shaders/3.1/light.vert", "../shaders/3.1/light.frag");
    auto line_shader = iris::shader_t::create("../shaders/3.1/line.vert", "../shaders/3.1/line.frag");

    // texture loading
    auto textures = std::vector<iris::texture_t>();
    textures.emplace_back(iris::texture_t::create("../textures/wall.jpg", iris::texture_type_t::non_linear_srgb));
    textures.emplace_back(iris::texture_t::create("../textures/container.png", iris::texture_type_t::non_linear_srgb));
    textures.emplace_back(iris::texture_t::create("../textures/container_specular.png", iris::texture_type_t::non_linear_srgb));

    auto meshes = std::vector<iris::mesh_t>();
    {
        auto cube_textures = std::vector<std::reference_wrapper<const iris::texture_t>>();
        cube_textures.emplace_back(std::cref(textures[1]));
        cube_textures.emplace_back(std::cref(textures[2]));
        meshes.emplace_back(iris::mesh_t::create(generate_cube(), {}, std::move(cube_textures)));
    }

    auto models = std::vector<iris::model_t>();
    models.emplace_back(iris::model_t::create("../models/deccer-cubes/SM_Deccer_Cubes_Textured.gltf"));
    // models.emplace_back(iris::model_t::create("../models/San_Miguel/san-miguel.obj"));
    // models.emplace_back(iris::model_t::create("../models/sponza/Sponza.gltf"));
    // models.emplace_back(iris::model_t::create("../models/chess/ABeautifulGame.gltf"));

    auto transforms = std::vector<glm::mat4>();
    transforms.emplace_back(glm::identity<glm::mat4>());
    transforms[0] = glm::scale(transforms[0], glm::vec3(0.75f));

    auto light_positions = std::vector<glm::vec3>();
    light_positions.emplace_back(-3.0f, 0.0f, 0.0f);
    light_positions.emplace_back(0.0f, 6.0f, 3.0f);
    light_positions.emplace_back(0.0f, 1.0f, -3.0f);
    light_positions.emplace_back(3.0f, 3.0f, 3.0f);

    auto light_transforms = std::vector<glm::mat4>();
    for (auto i = 0; i < 4; ++i) {
        auto transform = glm::identity<glm::mat4>();
        transform = glm::translate(transform, light_positions[i]);
        transform = glm::scale(transform, glm::vec3(0.1f));
        light_transforms.emplace_back(transform);
    }

    // aabb array section
    auto aabb_vao = 0_u32;
    auto aabb_vbo = 0_u32;

    glGenVertexArrays(1, &aabb_vao);
    glGenBuffers(1, &aabb_vbo);

    glBindVertexArray(aabb_vao);
    {
        const auto centered_cube = std::to_array({
            glm::vec3(-1.0f, -1.0f, -1.0f),
            glm::vec3( 1.0f, -1.0f, -1.0f),
            glm::vec3( 1.0f,  1.0f, -1.0f),
            glm::vec3(-1.0f,  1.0f, -1.0f),
            glm::vec3(-1.0f, -1.0f,  1.0f),
            glm::vec3( 1.0f, -1.0f,  1.0f),
            glm::vec3( 1.0f,  1.0f,  1.0f),
            glm::vec3(-1.0f,  1.0f,  1.0f),
        });
        const auto aabb_vertices = std::to_array({
            centered_cube[0], centered_cube[1],
            centered_cube[1], centered_cube[2],
            centered_cube[2], centered_cube[3],
            centered_cube[3], centered_cube[0],

            centered_cube[4], centered_cube[5],
            centered_cube[5], centered_cube[6],
            centered_cube[6], centered_cube[7],
            centered_cube[7], centered_cube[4],

            centered_cube[0], centered_cube[4],
            centered_cube[1], centered_cube[5],
            centered_cube[2], centered_cube[6],
            centered_cube[3], centered_cube[7],
        });
        glBindBuffer(GL_ARRAY_BUFFER, aabb_vbo);
        glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(aabb_vertices), aabb_vertices.data(), GL_STATIC_DRAW);
    }

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[3]), nullptr);

    // blending setup
    glEnable(GL_BLEND);
    // output.rgb = (input.rgb * input.a) + (previous.rgb * (1 - input.a))
    // output.a = (input.a * 1) + (previous.a * 0)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
    glBlendEquation(GL_FUNC_ADD);

    // face culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // framebuffers
    auto attachments = std::vector<iris::framebuffer_attachment_t>();
    attachments.emplace_back(iris::framebuffer_attachment_t::create(window.width, window.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE));
    attachments.emplace_back(iris::framebuffer_attachment_t::create(window.width, window.height, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8));

    auto framebuffer = iris::framebuffer_t::create({ std::cref(attachments[0]), std::cref(attachments[1]) });
    if (!framebuffer.is_complete()) {
        std::terminate();
    }

    // fullscreen quad
    auto f_quad_data = std::to_array({
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    });
    auto f_quad_vao = 0_u32;
    auto f_quad_vbo = 0_u32;
    glGenVertexArrays(1, &f_quad_vao);
    glGenBuffers(1, &f_quad_vbo);

    glBindVertexArray(f_quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, f_quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(f_quad_data), f_quad_data.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(iris::float32[4]), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(iris::float32[4]), reinterpret_cast<void*>(sizeof(iris::float32[2])));

    // timing
    auto delta_time = 0.0f;
    auto last_frame = 0.0f;

    // render loop
    glEnable(GL_SCISSOR_TEST);
    while (!glfwWindowShouldClose(window.handle)) {
        const auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_frame;
        last_frame = current_time;

        if (glfwGetKey(window.handle, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window.handle, true);
        }

        auto scene = scene_t();
        for (const auto& model : models) {
            for (const auto& mesh : model.meshes()) {
                auto is_opaque = true;
                for (const auto& texture : mesh.textures()) {
                    if (!texture.get().is_opaque()) {
                        is_opaque = false;
                        break;
                    }
                }
                if (is_opaque) {
                    scene.opaque_meshes.emplace_back(std::cref(mesh));
                } else {
                    scene.transparent_meshes.emplace_back(std::cref(mesh));
                }
            }
        }

        if (window.is_resized) {
            window.is_resized = false;
            attachments[0] = iris::framebuffer_attachment_t::create(window.width, window.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
            attachments[1] = iris::framebuffer_attachment_t::create(window.width, window.height, GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8);

            framebuffer = iris::framebuffer_t::create({ std::cref(attachments[0]), std::cref(attachments[1]) });
        }

        const auto lights_color = glm::normalize(glm::vec3(
            0.5f + (0.5f * glm::sin(current_time * 2.0f)),
            0.5f + (0.5f * glm::cos(current_time * 2.0f)),
            0.5f + (0.5f * glm::sin(current_time * 2.0f))));

        glEnable(GL_DEPTH_TEST);
        framebuffer.bind();

        // 1. render the frustum lines
        glScissor(0, 0, window.width, window.height);
        glViewport(0, 0, window.width, window.height);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // debug AABBs
        if (glfwGetKey(window.handle, GLFW_KEY_F) == GLFW_PRESS) {
            for (const auto& model : models) {
                for (const auto& mesh : model.meshes()) {
                    const auto& aabb = mesh.aabb();
                    auto transform = glm::identity<glm::mat4>();
                    transform = glm::translate(transform, aabb.center);
                    transform = glm::scale(transform, aabb.size / 2.0f);
                    transform = mesh.transform() * transform;

                    line_shader
                        .bind()
                        .set(0, camera.projection())
                        .set(1, camera.view())
                        .set(2, transform)
                        .set(3, { 1.0f, 1.0f, 1.0f });

                    glBindVertexArray(aabb_vao);
                    glDrawArrays(GL_LINES, 0, 24);
                }
            }
        }

        // 2. render opaque objects
        const auto projection = camera.projection();
        const auto view = camera.view();
        for (const auto& mesh : scene.opaque_meshes) {
            const auto& u_mesh = mesh.get();
            auto transform = u_mesh.transform();
            auto t_inv_transform = glm::inverseTranspose(transform);

            simple_shader
                .bind()
                .set(0, camera.projection())
                .set(1, camera.view())
                .set(2, transform)
                .set(3, t_inv_transform)
                .set(4, camera.position());
            for (auto j = 0_i32; const auto& texture : u_mesh.textures()) {
                texture.get().bind(j);
                simple_shader.set(5 + j, { j });
                j++;
            }
            simple_shader.set(7, { 32_u32 });

            for (auto j = 0_i32; j < 4; ++j) {
                simple_shader.set(8 + (7 * j), light_positions[j]);
                simple_shader.set(9 + (7 * j), glm::vec3(0.1f));
                simple_shader.set(10 + (7 * j), lights_color);
                simple_shader.set(11 + (7 * j), lights_color);
                simple_shader.set(12 + (7 * j), { 1.0f });
                simple_shader.set(13 + (7 * j), { 0.34f });
                simple_shader.set(14 + (7 * j), { 0.55f });
            }
            u_mesh.draw();
        }

        // sort and draw transparent objects
        std::sort(scene.transparent_meshes.begin(), scene.transparent_meshes.end(), [&camera](const auto& a, const auto& b) {
            const auto& a_mesh = a.get();
            const auto& b_mesh = b.get();
            const auto& a_aabb = a_mesh.aabb();
            const auto& b_aabb = b_mesh.aabb();

            const auto a_center = a_mesh.transform() * glm::vec4(a_aabb.center, 1.0f);
            const auto b_center = b_mesh.transform() * glm::vec4(b_aabb.center, 1.0f);
            const auto a_distance = glm::distance(camera.position(), glm::vec3(a_center));
            const auto b_distance = glm::distance(camera.position(), glm::vec3(b_center));
            return a_distance > b_distance;
        });
        for (const auto& mesh : scene.transparent_meshes) {
            const auto& u_mesh = mesh.get();
            auto transform = u_mesh.transform();
            auto t_inv_transform = glm::inverseTranspose(transform);

            simple_shader
                .bind()
                .set(0, camera.projection())
                .set(1, camera.view())
                .set(2, transform)
                .set(3, t_inv_transform)
                .set(4, camera.position());
            for (auto j = 0_i32; const auto& texture : u_mesh.textures()) {
                texture.get().bind(j);
                simple_shader.set(5 + j, { j });
                j++;
            }
            simple_shader.set(7, { 32_u32 });

            for (auto j = 0_i32; j < 4; ++j) {
                simple_shader.set(8 + (7 * j), light_positions[j]);
                simple_shader.set(9 + (7 * j), glm::vec3(0.1f));
                simple_shader.set(10 + (7 * j), lights_color);
                simple_shader.set(11 + (7 * j), lights_color);
                simple_shader.set(12 + (7 * j), { 1.0f });
                simple_shader.set(13 + (7 * j), { 0.34f });
                simple_shader.set(14 + (7 * j), { 0.55f });
            }
            u_mesh.draw();
        }

        for (auto i = 0_u32; i < 4; ++i) {
            light_shader
                .bind()
                .set(0, camera.projection())
                .set(1, camera.view())
                .set(2, light_transforms[i])
                .set(3, lights_color);

            meshes[0].draw();
        }

        // 3. render to the default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glScissor(0, 0, window.width, window.height);
        glViewport(0, 0, window.width, window.height);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 3.1. render the scene
        glDisable(GL_DEPTH_TEST);
        screen_shader
            .bind()
            .set(0, { 0 });
        glBindVertexArray(f_quad_vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, framebuffer.attachment(0).id());
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        // update the window (only cursor position for now) and camera.
        window.update();
        camera.update(delta_time);
    }
    return 0;
}
