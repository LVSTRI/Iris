#include <numeric>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <array>

#include <texture.hpp>
#include <shader.hpp>
#include <camera.hpp>
#include <utilities.hpp>
#include <mesh.hpp>
#include <model.hpp>

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

static auto generate_cube() noexcept -> std::vector<iris::vertex_t> {
    return std::vector<iris::vertex_t> {
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
    };
}

static auto calculate_frustum_lines(const glm::mat4& projection, const glm::mat4& view) -> std::vector<glm::vec3> {
    auto ndc_cube = std::vector<glm::vec4> {
        { -1.0f, -1.0f, -1.0f, 1.0f },
        {  1.0f, -1.0f, -1.0f, 1.0f },
        {  1.0f,  1.0f, -1.0f, 1.0f },
        { -1.0f,  1.0f, -1.0f, 1.0f },
        { -1.0f, -1.0f,  1.0f, 1.0f },
        {  1.0f, -1.0f,  1.0f, 1.0f },
        {  1.0f,  1.0f,  1.0f, 1.0f },
        { -1.0f,  1.0f,  1.0f, 1.0f },
    };
    for (auto& v : ndc_cube) {
        v = glm::inverse(projection * view) * v;
        v /= v.w;
    }

    return std::vector<glm::vec3> {
        ndc_cube[0], ndc_cube[1],
        ndc_cube[1], ndc_cube[2],
        ndc_cube[2], ndc_cube[3],
        ndc_cube[3], ndc_cube[0],

        ndc_cube[4], ndc_cube[5],
        ndc_cube[5], ndc_cube[6],
        ndc_cube[6], ndc_cube[7],
        ndc_cube[7], ndc_cube[4],

        ndc_cube[0], ndc_cube[4],
        ndc_cube[1], ndc_cube[5],
        ndc_cube[2], ndc_cube[6],
        ndc_cube[3], ndc_cube[7],
    };
}

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
    auto simple_shader = iris::shader_t::create("../shaders/2.1/simple.vert", "../shaders/2.1/simple.frag");
    auto light_shader = iris::shader_t::create("../shaders/2.1/light.vert", "../shaders/2.1/light.frag");
    auto frustum_shader = iris::shader_t::create("../shaders/2.1/frustum.vert", "../shaders/2.1/frustum.frag");
    auto line_shader = iris::shader_t::create("../shaders/2.1/line.vert", "../shaders/2.1/line.frag");

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

    // enable depth testing.
    glEnable(GL_DEPTH_TEST);

    // frustum array section
    auto frustum_lines = calculate_frustum_lines(camera.projection(), camera.view());

    auto frustum_vao = 0_u32;
    auto frustum_vbo = 0_u32;

    glGenVertexArrays(1, &frustum_vao);
    glGenBuffers(1, &frustum_vbo);

    glBindVertexArray(frustum_vao);

    glBindBuffer(GL_ARRAY_BUFFER, frustum_vbo);
    glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(frustum_lines), frustum_lines.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(iris::float32[3]), nullptr);

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

    // timing.
    auto delta_time = 0.0f;
    auto last_frame = 0.0f;

    const auto* hit_mesh = std::type_identity_t<const iris::mesh_t*>();

    bool is_mouse_pressed[2] = {
        static_cast<bool>(glfwGetMouseButton(window.handle, GLFW_MOUSE_BUTTON_LEFT)),
        static_cast<bool>(glfwGetMouseButton(window.handle, GLFW_MOUSE_BUTTON_LEFT)),
    };

    // render loop
    glEnable(GL_SCISSOR_TEST);
    while (!glfwWindowShouldClose(window.handle)) {
        const auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_frame;
        last_frame = current_time;

        is_mouse_pressed[0] = is_mouse_pressed[1];
        is_mouse_pressed[1] = static_cast<bool>(glfwGetMouseButton(window.handle, GLFW_MOUSE_BUTTON_LEFT));


        if (glfwGetKey(window.handle, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window.handle, true);
        }

        if (!window.is_mouse_captured && window.is_focused) {
            auto c_x = 0.0;
            auto c_y = 0.0;
            glfwGetCursorPos(window.handle, &c_x, &c_y);
            const auto is_oob =
                c_x < 0.0 || c_x > window.width ||
                c_y < 0.0 || c_y > window.height;
            if (!is_oob && is_mouse_pressed[0]) {
                hit_mesh = nullptr;
                // NDC mouse position
                const auto ndc_cursor = glm::vec2(
                    (2.0f * c_x) / static_cast<iris::float32>(window.width) - 1.0f,
                    1.0f - (2.0f * c_y) / static_cast<iris::float32>(window.height));
                const auto projection = camera.projection();
                const auto view = camera.view();
                // NDC -> world space
                const auto ndc_near = glm::vec4(ndc_cursor, -1.0f, 1.0f);
                const auto ndc_far = glm::vec4(ndc_cursor, 1.0f, 1.0f);
                auto world_near = glm::inverse(projection * view) * ndc_near;
                auto world_far = glm::inverse(projection * view) * ndc_far;
                world_near /= world_near.w;
                world_far /= world_far.w;

                // ray construction
                const auto ray_origin = glm::vec3(world_near);
                const auto ray_direction = glm::vec3(world_far) - glm::vec3(world_near);

                auto t_best = std::numeric_limits<iris::float32>::max();

                for (const auto& model : models) {
                    for (const auto& mesh : model.meshes()) {
                        // transform the aabb extents to world space
                        const auto& aabb = mesh.aabb();
                        const auto world_aabb_min = mesh.transform() * glm::vec4(aabb.min, 1.0f);
                        const auto world_aabb_max = mesh.transform() * glm::vec4(aabb.max, 1.0f);

                        auto t_min = 0.0f;
                        auto t_max = std::numeric_limits<iris::float32>::infinity();
                        for (auto i = 0_i32; i < 3; ++i) {
                            // intersect
                            const auto inv_dir = 1 / ray_direction[i];
                            const auto t1 = (world_aabb_min[i] - ray_origin[i]) * inv_dir;
                            const auto t2 = (world_aabb_max[i] - ray_origin[i]) * inv_dir;

                            t_min = glm::min(glm::max(t1, t_min), glm::max(t2, t_min));
                            t_max = glm::max(glm::min(t1, t_max), glm::min(t2, t_max));
                        }
                        if (t_max >= 0 && t_min > 0 && t_min <= t_max && t_best > t_min) {
                            // we have a hit
                            hit_mesh = &mesh;
                            t_best = t_min;
                        }
                    }
                }
            }
        }

        // 1. render the frustum lines
        glScissor(0, 0, window.width, window.height);
        glViewport(0, 0, window.width, window.height);
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(frustum_vao);
        // 1.5. update the frustum if requested
        if (glfwGetKey(window.handle, GLFW_KEY_F1) == GLFW_PRESS) {
            frustum_lines = calculate_frustum_lines(camera.projection(), camera.view());

            glBindBuffer(GL_ARRAY_BUFFER, frustum_vbo);
            glBufferData(GL_ARRAY_BUFFER, iris::size_bytes(frustum_lines), frustum_lines.data(), GL_STATIC_DRAW);
        }

        frustum_shader
            .bind()
            .set(0, camera.projection())
            .set(1, camera.view())
            .set(2, glm::identity<glm::mat4>())
            .set(3, { 1.0f, 0.0f, 0.0f });
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 24);

        // debug AABBs
        if (glfwGetKey(window.handle, GLFW_KEY_F) == GLFW_PRESS) {
            for (const auto& model : models) {
                for (const auto& mesh : model.meshes()) {
                    const auto& aabb = mesh.aabb();
                    auto transform = glm::identity<glm::mat4>();
                    transform = glm::translate(transform, aabb.center);
                    transform = glm::scale(transform, aabb.size / 2.0f);
                    transform = mesh.transform() * transform;

                    frustum_shader
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

        glLineWidth(8.0f);
        if (hit_mesh) {
        //if (!hit_meshes.empty()) {
            //for (const auto& hit_mesh : hit_meshes) {
                const auto& u_mesh = *hit_mesh;
                const auto& aabb = u_mesh.aabb();
                auto transform = glm::identity<glm::mat4>();
                transform = glm::translate(transform, aabb.center);
                transform = glm::scale(transform, aabb.size / 2.0f);
                transform = u_mesh.transform() * transform;

                frustum_shader
                    .bind()
                    .set(0, camera.projection())
                    .set(1, camera.view())
                    .set(2, transform)
                    .set(3, { 1.0f, 1.0f, 1.0f });

                glBindVertexArray(aabb_vao);
                glDrawArrays(GL_LINES, 0, 24);
            //}
        //}
        }

        // 2. render the scene normally
        const auto projection = camera.projection();
        const auto view = camera.view();

        for (auto i = 0_u32; const auto& model : models) {
            for (const auto& mesh : model.meshes()) {
                auto transform = mesh.transform();
                auto t_inv_transform = glm::inverseTranspose(transform);

                simple_shader
                    .bind()
                    .set(0, camera.projection())
                    .set(1, camera.view())
                    .set(2, transform)
                    .set(3, t_inv_transform)
                    .set(4, camera.position());
                for (auto j = 0_i32; const auto& texture : mesh.textures()) {
                    texture.get().bind(j);
                    simple_shader.set(5 + j, { j });
                    j++;
                }
                simple_shader.set(7, { 32_u32 });

                for (auto j = 0_i32; j < 4; ++j) {
                    const auto color = glm::normalize(glm::vec3(
                        0.5f + (0.5f * glm::sin(current_time * 2.0f)),
                        0.5f + (0.5f * glm::cos(current_time * 2.0f)),
                        0.5f + (0.5f * glm::sin(current_time * 2.0f))));
                    simple_shader.set(8 + (7 * j), light_positions[j]);
                    simple_shader.set(9 + (7 * j), glm::vec3(0.1f));
                    simple_shader.set(10 + (7 * j), color);
                    simple_shader.set(11 + (7 * j), color);
                    simple_shader.set(12 + (7 * j), { 1.0f });
                    simple_shader.set(13 + (7 * j), { 0.34f });
                    simple_shader.set(14 + (7 * j), { 0.55f });
                }
                mesh.draw();
            }
            i++;
        }

        for (auto i = 0_u32; i < 4; ++i) {
            const auto color = glm::normalize(glm::vec3(
                0.5f + (0.5f * glm::sin(current_time * 2.0f)),
                0.5f + (0.5f * glm::cos(current_time * 2.0f)),
                0.5f + (0.5f * glm::sin(current_time * 2.0f))));

            light_shader
                .bind()
                .set(0, camera.projection())
                .set(1, camera.view())
                .set(2, light_transforms[i])
                .set(3, color);

            meshes[0].draw();
        }

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        // update the window (only cursor position for now) and camera.
        window.update();
        camera.update(delta_time);
    }

    return 0;
}
