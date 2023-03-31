#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <array>
#include <functional>
#include <optional>

#include <texture.hpp>
#include <shader.hpp>
#include <camera.hpp>
#include <utilities.hpp>
#include <mesh.hpp>
#include <model.hpp>
#include <framebuffer.hpp>
#include <buffer.hpp>

#include <debug_break.hpp>

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

struct camera_data_t {
    glm::mat4 projection;
    glm::mat4 view;
    glm::vec3 position;
};

struct point_light_t {
    glm::vec3 position = {};
    iris::float32 _pad0 = 0;
    glm::vec3 ambient = {};
    iris::float32 _pad1 = 0;
    glm::vec3 diffuse = {};
    iris::float32 _pad2 = 0;
    glm::vec3 specular = {};
    iris::float32 constant = 0;
    iris::float32 linear = 0;
    iris::float32 quadratic = 0;
    iris::float32 _pad4[2] = {};
};

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

struct mesh_ref_t {
    std::reference_wrapper<const iris::mesh_t> ref;
    iris::uint32 id = 0;
};

struct scene_t {
    std::vector<mesh_ref_t> opaque_meshes;
    std::vector<mesh_ref_t> transparent_meshes;
    std::vector<mesh_ref_t> meshes;
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
            debug_break();
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
    auto fullscreen_shader = iris::shader_t::create("../shaders/4.1/fullscreen.vert", "../shaders/4.1/fullscreen.frag");
    auto simple_shader = iris::shader_t::create("../shaders/4.1/simple.vert", "../shaders/4.1/simple.frag");
    auto light_shader = iris::shader_t::create("../shaders/4.1/light.vert", "../shaders/4.1/light.frag");
    auto line_shader = iris::shader_t::create("../shaders/4.1/line.vert", "../shaders/4.1/line.frag");

    // texture loading
    auto textures = std::vector<iris::texture_t>();
    textures.emplace_back(iris::texture_t::create("../textures/wall.jpg"));
    textures.emplace_back(iris::texture_t::create("../textures/container.png"));
    textures.emplace_back(iris::texture_t::create("../textures/container_specular.png"));

    auto meshes = std::vector<iris::mesh_t>();
    {
        auto cube_textures = std::vector<std::reference_wrapper<const iris::texture_t>>();
        cube_textures.emplace_back(std::cref(textures[1]));
        cube_textures.emplace_back(std::cref(textures[2]));
        meshes.emplace_back(iris::mesh_t::create(generate_cube(), {}, std::move(cube_textures)));
    }

    auto models = std::vector<iris::model_t>();
    // models.emplace_back(iris::model_t::create("../models/deccer-cubes/SM_Deccer_Cubes_Textured.gltf"));
    // models.emplace_back(iris::model_t::create("../models/San_Miguel/san-miguel.obj"));
    models.emplace_back(iris::model_t::create("../models/sponza/Sponza.gltf"));
    // models.emplace_back(iris::model_t::create("../models/chess/ABeautifulGame.gltf"));

    auto transforms = std::vector<std::array<glm::mat4, 2>>();
    for (const auto& model : models) {
        for (const auto& mesh : model.meshes()) {
            transforms.emplace_back(std::to_array({ mesh.transform(), glm::inverseTranspose(mesh.transform())} ));
        }
    }

    auto light_positions = std::vector<glm::vec3>();
    light_positions.emplace_back( 0.0f, 0.5f,  0.0f);
    light_positions.emplace_back( 3.0f, 0.5f,  0.0f);
    light_positions.emplace_back( 3.0f, 0.5f,  3.0f);
    light_positions.emplace_back( 3.0f, 0.5f, -3.0f);
    light_positions.emplace_back(-3.0f, 2.5f,  3.0f);
    light_positions.emplace_back(-3.0f, 2.5f, -3.0f);
    light_positions.emplace_back( 3.0f, 2.5f,  3.0f);
    light_positions.emplace_back( 6.0f, 0.5f,  3.0f);
    light_positions.emplace_back( 6.0f, 0.5f, -3.0f);
    light_positions.emplace_back(-6.0f, 0.5f,  3.0f);
    light_positions.emplace_back(-6.0f, 0.5f, -3.0f);


    auto light_transforms = std::vector<glm::mat4>();
    for (const auto& light_position : light_positions) {
        auto transform = glm::identity<glm::mat4>();
        transform = glm::translate(transform, light_position);
        transform = glm::scale(transform, glm::vec3(0.1f));
        light_transforms.emplace_back(transform);
    }

    auto point_lights = std::vector<point_light_t>();
    for (const auto& light_position : light_positions) {
        const auto color = glm::normalize(0.25f + glm::vec3(
            rand() / static_cast<iris::float32>(RAND_MAX),
            rand() / static_cast<iris::float32>(RAND_MAX),
            rand() / static_cast<iris::float32>(RAND_MAX)));
        point_lights.emplace_back(point_light_t{
            .position = light_position,
            .ambient = glm::vec3(0.1f),
            .diffuse = color,
            .specular = color,
            .constant = 1.0f,
            .linear = 0.34f,
            .quadratic = 0.44f,
        });
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

    // multisampling
    glEnable(GL_SAMPLE_ALPHA_TO_ONE);
    glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);

    // mouse picking
    auto f0_main_attachments = std::to_array({
        iris::framebuffer_attachment_t::create(
            window.width,
            window.height,
            GL_RGBA8,
            GL_RGBA,
            GL_UNSIGNED_BYTE),
        iris::framebuffer_attachment_t::create(
            window.width,
            window.height,
            GL_R32UI,
            GL_RED_INTEGER,
            GL_UNSIGNED_INT),
        iris::framebuffer_attachment_t::create(
            window.width,
            window.height,
            GL_DEPTH24_STENCIL8,
            GL_DEPTH_STENCIL,
            GL_UNSIGNED_INT_24_8),
    });

    auto f0_main = iris::framebuffer_t::create({
        std::cref(f0_main_attachments[0]),
        std::cref(f0_main_attachments[1]),
        std::cref(f0_main_attachments[2]),
    });
    {
        const auto draw_attachments = std::to_array<GLenum>({
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1
        });
        glDrawBuffers(2, draw_attachments.data());
        glReadBuffer(GL_COLOR_ATTACHMENT1);
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

    // uniform buffers
    auto camera_buffer = iris::buffer_t::create(sizeof(camera_data_t), GL_UNIFORM_BUFFER);
    auto model_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto point_light_buffer = iris::buffer_t::create(sizeof(point_light_t[16]), GL_SHADER_STORAGE_BUFFER);

    // raycasting
    auto hit_mesh = std::optional<std::pair<std::reference_wrapper<const iris::mesh_t>, iris::uint32>>();

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
        {
            auto mesh_id = 0_u32;
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
                        scene.opaque_meshes.push_back({ std::cref(mesh), mesh_id });
                    } else {
                        scene.transparent_meshes.push_back({ std::cref(mesh), mesh_id });
                    }
                    scene.meshes.push_back({ std::cref(mesh), mesh_id });
                    mesh_id++;
                }
            }
        }

        if (window.is_resized) {
            f0_main_attachments = std::to_array({
                iris::framebuffer_attachment_t::create(
                    window.width,
                    window.height,
                    GL_RGBA8,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE),
                iris::framebuffer_attachment_t::create(
                    window.width,
                    window.height,
                    GL_R32UI,
                    GL_RED_INTEGER,
                    GL_UNSIGNED_INT),
                iris::framebuffer_attachment_t::create(
                    window.width,
                    window.height,
                    GL_DEPTH24_STENCIL8,
                    GL_DEPTH_STENCIL,
                    GL_UNSIGNED_INT_24_8),
            });

            f0_main = iris::framebuffer_t::create({
                std::cref(f0_main_attachments[0]),
                std::cref(f0_main_attachments[1]),
                std::cref(f0_main_attachments[2]),
            });

            const auto draw_attachments = std::to_array<GLenum>({
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1
            });
            glDrawBuffers(2, draw_attachments.data());
            glReadBuffer(GL_COLOR_ATTACHMENT1);
            window.is_resized = false;
        }

        auto camera_data = camera_data_t {
            camera.projection(),
            camera.view(),
            camera.position()
        };

        camera_buffer.write(&camera_data, iris::size_bytes(camera_data));
        model_buffer.write(transforms.data(), iris::size_bytes(transforms));
        point_light_buffer.write(point_lights.data(), iris::size_bytes(point_lights));

        // 1. render the frustum lines
        f0_main.bind();
        glEnable(GL_DEPTH_TEST);
        glScissor(0, 0, window.width, window.height);
        glViewport(0, 0, window.width, window.height);
        const auto clear_color = glm::vec4(0.05f, 0.05f, 0.05f, 1.0f);
        const auto clear_id = glm::uvec4(0xffffffff);
        const auto clear_depth = glm::vec4(1.0f);
        glClearBufferfv(GL_COLOR, 0, glm::value_ptr(clear_color));
        glClearBufferuiv(GL_COLOR, 1, glm::value_ptr(clear_id));
        glClearBufferfv(GL_DEPTH, 0, glm::value_ptr(clear_depth));

        // debug AABBs
        if (glfwGetKey(window.handle, GLFW_KEY_F) == GLFW_PRESS) {
            auto mesh_id = 0_u32;
            for (const auto& model : models) {
                for (const auto& mesh : model.meshes()) {
                    const auto& aabb = mesh.aabb();
                    auto transform = glm::identity<glm::mat4>();
                    transform = glm::translate(transform, aabb.center);
                    transform = glm::scale(transform, aabb.size / 2.0f);
                    transform = transforms[mesh_id][0] * transform;

                    line_shader
                        .bind()
                        .set(0, transform)
                        .set(3, { 1.0f, 1.0f, 1.0f });

                    camera_buffer.bind_base(0);

                    glBindVertexArray(aabb_vao);
                    glDrawArrays(GL_LINES, 0, 24);
                    mesh_id++;
                }
            }
        }

        // 2. render opaque objects
        const auto projection = camera.projection();
        const auto view = camera.view();
        for (const auto& mesh : scene.opaque_meshes) {
            const auto& u_mesh = mesh.ref.get();

            simple_shader
                .bind()
                .set(0, { mesh.id });

            camera_buffer.bind_base(0);
            model_buffer.bind_range(1, 0, iris::size_bytes(transforms));
            point_light_buffer.bind_range(2, 0, iris::size_bytes(point_lights));

            for (auto j = 0_i32; const auto& texture : u_mesh.textures()) {
                texture.get().bind(j);
                simple_shader.set(4 + j, { j });
                j++;
            }
            simple_shader.set(6, { 32_u32 });
            simple_shader.set(7, { static_cast<iris::uint32>(point_lights.size()) });

            u_mesh.draw();
        }

        // sort and draw transparent objects
        auto indices = std::vector<std::size_t>(scene.transparent_meshes.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&camera, &transforms, &scene](const auto& i, const auto& j) {
            const auto& a_mesh = scene.transparent_meshes[i].ref.get();
            const auto& b_mesh = scene.transparent_meshes[j].ref.get();
            const auto& a_aabb = a_mesh.aabb();
            const auto& b_aabb = b_mesh.aabb();

            const auto a_center = transforms[i][0] * glm::vec4(a_aabb.center, 1.0f);
            const auto b_center = transforms[j][0] * glm::vec4(b_aabb.center, 1.0f);
            const auto a_distance = glm::distance(camera.position(), glm::vec3(a_center));
            const auto b_distance = glm::distance(camera.position(), glm::vec3(b_center));
            return a_distance > b_distance;
        });
        for (const auto& mesh : scene.transparent_meshes) {
            const auto& u_mesh = mesh.ref.get();

            simple_shader
                .bind()
                .set(0, { mesh.id });

            camera_buffer.bind_base(0);
            model_buffer.bind_range(1, 0, iris::size_bytes(transforms));
            point_light_buffer.bind_range(2, 0, iris::size_bytes(point_lights));

            for (auto j = 0_i32; const auto& texture : u_mesh.textures()) {
                texture.get().bind(j);
                simple_shader.set(4 + j, { j });
                j++;
            }
            simple_shader.set(6, { 32_u32 });
            simple_shader.set(7, { static_cast<iris::uint32>(point_lights.size()) });

            u_mesh.draw();
        }

        for (auto i = 0_u32; i < light_positions.size(); ++i) {
            light_shader
                .bind()
                .set(0, light_transforms[i])
                .set(3, point_lights[i].diffuse);

            camera_buffer.bind_base(0);

            meshes[0].draw();
        }

        // highlight the hit object (if any)
        if (hit_mesh) {
            const auto& [r_mesh, id] = *hit_mesh;
            const auto& aabb = r_mesh.get().aabb();
            auto transform = glm::identity<glm::mat4>();
            transform = glm::translate(transform, aabb.center);
            transform = glm::scale(transform, aabb.size / 2.0f);
            transform = transforms[id][0] * transform;

            line_shader
                .bind()
                .set(0, transform)
                .set(3, { 1.0f, 1.0f, 1.0f });

            camera_buffer.bind_base(0);

            glBindVertexArray(aabb_vao);
            glDrawArrays(GL_LINES, 0, 24);
        }

        // read the mesh_id buffer if anything was clicked
        if (window.is_focused) {
            auto c_x = 0.0;
            auto c_y = 0.0;
            glfwGetCursorPos(window.handle, &c_x, &c_y);
            const auto is_oob =
                c_x < 0.0 || c_x > window.width ||
                c_y < 0.0 || c_y > window.height;
            if (!is_oob && glfwGetMouseButton(window.handle, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                auto mesh_id = -1_u32;
                glReadPixels(c_x, window.height - c_y, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &mesh_id);
                if (mesh_id != -1 && mesh_id < scene.meshes.size()) {
                    hit_mesh = {
                        std::ref(scene.meshes[mesh_id].ref.get()),
                        mesh_id
                    };
                }
            }
        }

        // render to default framebuffer
        fullscreen_shader.bind();
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, window.width, window.height);
        glScissor(0, 0, window.width, window.height);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray(f_quad_vao);
        glBindTexture(GL_TEXTURE_2D, f0_main.attachment(0).id());
        fullscreen_shader.set(0, { 0_i32 });
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        // update the window (only cursor position for now) and camera.
        window.update();
        camera.update(delta_time);
    }
    return 0;
}
