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

#define CASCADE_COUNT 4

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

struct directional_light_t {
    glm::vec3 direction = {};
    iris::float32 _pad0 = 0;
    glm::vec3 diffuse = {};
    iris::float32 _pad1 = 0;
    glm::vec3 specular = {};
    iris::float32 _pad2 = 0;
};

struct cascade_setup_data_t {
    glm::mat4 global_shadow_pv = {};
    glm::mat4 inv_pv = {};
    glm::vec4 camera_right = {};
    glm::vec4 light_dir = {};
    iris::float32 near = {};
    iris::float32 far = {};
    iris::float32 shadow_size = {};
};

struct cascade_data_t {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 pv;
    glm::mat4 global;
    glm::vec4 scale;
    glm::vec4 offset; // w is split
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

static auto calculate_shadow_frustum(const iris::camera_t& camera, const glm::vec3 light_dir) noexcept -> std::pair<std::vector<cascade_data_t>, glm::mat4> {
    const auto partition = [&](iris::float32 near, iris::float32 far) {
        auto shadow_frustum = cascade_data_t();
        const auto ndc_cube = std::to_array({
            glm::vec3(-1.0f, -1.0f, -1.0f),
            glm::vec3( 1.0f, -1.0f, -1.0f),
            glm::vec3(-1.0f,  1.0f, -1.0f),
            glm::vec3( 1.0f,  1.0f, -1.0f),
            glm::vec3(-1.0f, -1.0f,  1.0f),
            glm::vec3( 1.0f, -1.0f,  1.0f),
            glm::vec3(-1.0f,  1.0f,  1.0f),
            glm::vec3( 1.0f,  1.0f,  1.0f),
        });

        const auto c_proj = glm::perspective(camera.fov(), camera.aspect(), near, far);
        const auto inv_pv = glm::inverse(c_proj * camera.view());
        auto world_points = std::vector<glm::vec3>();
        world_points.reserve(ndc_cube.size());
        for (const auto& point : ndc_cube) {
            auto world_point = inv_pv * glm::vec4(point, 1.0f);
            world_point /= world_point.w;
            world_points.emplace_back(world_point);
        }

        // frustum center
        auto center = glm::vec3(0.0f);
        for (const auto& point : world_points) {
            center += point;
        }
        center /= world_points.size();

        // world -> light view space
        const auto light_view = glm::lookAt(center + light_dir * 0.5f, center, glm::vec3(0.0f, 1.0f, 0.0f));

        // calculate frustum bounding box in light view space
        auto min = glm::vec3(std::numeric_limits<iris::float32>::max());
        auto max = glm::vec3(std::numeric_limits<iris::float32>::lowest());
        for (const auto& point : world_points) {
            const auto light_space_point = glm::vec3(light_view * glm::vec4(point, 1.0f));
            min = glm::min(min, light_space_point);
            max = glm::max(max, light_space_point);
        }
        if (min.z < 0) {
            min.z *= 15.0f;
        } else {
            min.z /= 15.0f;
        }
        if (max.z < 0) {
            max.z /= 10.0f;
        } else {
            max.z *= 10.0f;
        }

        // light projection
        shadow_frustum.projection = glm::ortho(min.x, max.x, min.y, max.y, min.z, max.z);
        shadow_frustum.view = light_view;
        shadow_frustum.pv = shadow_frustum.projection * shadow_frustum.view;
        shadow_frustum.offset = { 0.0f, 0.0f, near, far };

        return shadow_frustum;
    };
    const auto make_global_matrix = [&]() {
        auto shadow_frustum = cascade_data_t();
        const auto ndc_cube = std::to_array({
            glm::vec3(-1.0f, -1.0f, -1.0f),
            glm::vec3( 1.0f, -1.0f, -1.0f),
            glm::vec3(-1.0f,  1.0f, -1.0f),
            glm::vec3( 1.0f,  1.0f, -1.0f),
            glm::vec3(-1.0f, -1.0f,  1.0f),
            glm::vec3( 1.0f, -1.0f,  1.0f),
            glm::vec3(-1.0f,  1.0f,  1.0f),
            glm::vec3( 1.0f,  1.0f,  1.0f),
        });

        const auto inv_pv = glm::inverse(camera.projection() * camera.view());

        auto frustum_center = glm::vec3(0.0f);
        auto world_points = std::vector<glm::vec3>();
        world_points.reserve(ndc_cube.size());
        for (const auto& point : ndc_cube) {
            auto world_point = inv_pv * glm::vec4(point, 1.0f);
            world_point /= world_point.w;
            world_points.emplace_back(world_point);
            frustum_center += glm::vec3(world_point);
        }
        frustum_center /= 8.0f;

        auto min_ext = glm::vec3(std::numeric_limits<iris::float32>::max());
        auto max_ext = glm::vec3(std::numeric_limits<iris::float32>::lowest());
        for (auto i = 0_u32; i < 8; ++i) {
            min_ext = glm::vec3(std::min(min_ext.x, world_points[i].x));
            min_ext = glm::vec3(std::min(min_ext.y, world_points[i].y));
            min_ext = glm::vec3(std::min(min_ext.z, world_points[i].z));
            max_ext = glm::vec3(std::max(max_ext.x, world_points[i].x));
            max_ext = glm::vec3(std::max(max_ext.y, world_points[i].y));
            max_ext = glm::vec3(std::max(max_ext.z, world_points[i].z));
        }
        shadow_frustum.projection = glm::ortho(min_ext.x, max_ext.x, min_ext.y, max_ext.y, 0.0f, 1.0f);
        shadow_frustum.view = glm::lookAt(frustum_center + light_dir * 0.5f, frustum_center, glm::vec3(0.0f, 1.0f, 0.0f));
        shadow_frustum.pv = shadow_frustum.projection * shadow_frustum.view;
        return shadow_frustum;
    };
    const auto global = make_global_matrix();
    const auto uv_scale_bias = glm::mat4(
        glm::vec4(0.5f, 0.0f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 0.0f),
        glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
    return std::make_pair(std::vector{
        partition(camera.near(), 5.0f),
        partition(5.0f, 10.0f),
        partition(10.0f, 20.0f),
        partition(20.0f, camera.far()),
    }, uv_scale_bias * global.pv);
}

static auto calculate_pssm_cascades(const iris::camera_t& camera, const glm::mat4& global_pv, glm::vec3 light_dir, iris::float32 resolution) noexcept -> std::vector<cascade_data_t> {
    const auto partition = [&](iris::uint32 cascade_index) {
        const auto near_clip = camera.near();
        const auto far_clip = camera.far();

        const auto min_depth = 0.0f;
        const auto max_depth = 1.0f;

        auto cascade_splits = std::to_array<iris::float32, CASCADE_COUNT>({});
        {
            const auto lambda = 1.0f;
            const auto clip_range = far_clip - near_clip;
            const auto min_z = near_clip + min_depth * clip_range;
            const auto max_z = near_clip + max_depth * clip_range;
            const auto range = max_z - min_z;
            const auto ratio = max_z / min_z;

            for (auto i = 0_u32; i < CASCADE_COUNT; ++i) {
                const auto p = (i + 1) / static_cast<iris::float32>(CASCADE_COUNT);
                const auto s_log = min_z * glm::pow(ratio, p);
                const auto s_uniform = min_z + range * p;
                const auto d = lambda * (s_log - s_uniform) + s_uniform;
                cascade_splits[i] = (d - near_clip) / clip_range;
            }
        }

        auto frustum_corners = std::to_array({
            glm::vec3(-1.0f, -1.0f, -1.0f),
            glm::vec3( 1.0f, -1.0f, -1.0f),
            glm::vec3(-1.0f,  1.0f, -1.0f),
            glm::vec3( 1.0f,  1.0f, -1.0f),
            glm::vec3(-1.0f, -1.0f,  1.0f),
            glm::vec3( 1.0f, -1.0f,  1.0f),
            glm::vec3(-1.0f,  1.0f,  1.0f),
            glm::vec3( 1.0f,  1.0f,  1.0f),
        });

        const auto prev_split = cascade_index == 0 ? min_depth : cascade_splits[cascade_index - 1];
        const auto curr_split = cascade_splits[cascade_index];
        const auto inv_pv = glm::inverse(camera.projection() * camera.view());
        for (auto i = 0_u32; i < 8; ++i) {
            const auto corner = inv_pv * glm::vec4(frustum_corners[i], 1.0f);
            frustum_corners[i] = corner / corner.w;
        }

        for (auto i = 0_u32; i < 4; ++i) {
            const auto corner_ray = frustum_corners[i + 4] - frustum_corners[i];
            frustum_corners[i + 4] = frustum_corners[i] + (corner_ray * prev_split);
            frustum_corners[i] = frustum_corners[i] + (corner_ray * curr_split);
        }

        auto frustum_center = glm::vec3(0.0f);
        for (auto i = 0_u32; i < 8; ++i) {
            frustum_center += frustum_corners[i];
        }
        frustum_center /= 8.0f;

        const auto d_up = glm::vec3(0.0f, 1.0f, 0.0f);

        auto min_ext = glm::vec3(0.0f);
        auto max_ext = glm::vec3(0.0f);
        auto r_sphere = 0.0f;
        for (auto i = 0_u32; i < 8; ++i) {
            r_sphere = glm::max(r_sphere, glm::length(frustum_corners[i] - frustum_center));
        }
        max_ext = glm::vec3(r_sphere);
        min_ext = -max_ext;

        const auto cascade_ext = max_ext - min_ext;
        const auto light_cam_pos = frustum_center + light_dir * -min_ext.z;
        auto light_view = glm::lookAt(light_cam_pos, frustum_center, d_up);
        auto light_proj = glm::ortho(min_ext.x, max_ext.x, min_ext.y, max_ext.y, 0.0f, cascade_ext.z);

        // stabilize
        {
            const auto light_pv = light_proj * light_view;
            auto shadow_origin = glm::vec3(0.0f);
            shadow_origin = glm::vec3(light_pv * glm::vec4(shadow_origin, 1.0));
            shadow_origin *= resolution / 2.0;
            const auto rounded_origin = glm::round(shadow_origin);
            auto round_offset = rounded_origin - shadow_origin;
            round_offset *= 2.0 / resolution;
            light_proj[3][0] += round_offset.x;
            light_proj[3][1] += round_offset.y;
        }
        const auto light_pv = light_proj * light_view;

        auto cascade_data = cascade_data_t();
        cascade_data.projection = light_proj;
        cascade_data.view = light_view;
        cascade_data.pv = light_pv;

        const auto clip_dist = far_clip - near_clip;

        auto uv_scale_bias = glm::mat4(
            glm::vec4(0.5f, 0.0f, 0.0f, 0.0f),
            glm::vec4(0.0f, 0.5f, 0.0f, 0.0f),
            glm::vec4(0.0f, 0.0f, 0.5f, 0.0f),
            glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
        const auto inv_cascade = glm::inverse(uv_scale_bias * (light_proj * light_view));

        auto cascade_corners = std::to_array({
            glm::vec3(inv_cascade * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)),
            glm::vec3(inv_cascade * glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)),
        });
        cascade_corners[0] = glm::vec3(global_pv * glm::vec4(cascade_corners[0], 1.0f));
        cascade_corners[1] = glm::vec3(global_pv * glm::vec4(cascade_corners[1], 1.0f));

        const auto cascade_scale = 1.0f / (cascade_corners[1] - cascade_corners[0]);

        cascade_data.offset = glm::vec4(-cascade_corners[0], near_clip + curr_split * clip_dist);
        cascade_data.scale = glm::make_vec4(cascade_scale);
        cascade_data.global = global_pv;

        return cascade_data;
    };
    return {
        partition(0),
        partition(1),
        partition(2),
        partition(3),
    };
}

static auto calculate_workgroup_count_from_wh(iris::uint32 width, iris::uint32 height) noexcept -> std::vector<glm::uvec2> {
    constexpr auto workgroup_size = 16_u32;
    auto workgroup_count = std::vector<glm::uvec2>();
    workgroup_count.reserve(16);
    workgroup_count.emplace_back(
        (width + workgroup_size - 1) / workgroup_size,
        (height + workgroup_size - 1) / workgroup_size);
    while (workgroup_count.back() != glm::uvec2(1)) {
        const auto& previous = workgroup_count.back();
        const auto current_size = glm::uvec2(
            (previous.x + workgroup_size - 1) / workgroup_size,
            (previous.y + workgroup_size - 1) / workgroup_size);
        workgroup_count.emplace_back(glm::max(current_size, glm::uvec2(1)));
    }
    return workgroup_count;
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
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

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
    auto shadow_shader = iris::shader_t::create("../shaders/4.1/shadow.vert", "../shaders/4.1/empty.frag");
    auto depth_only_shader = iris::shader_t::create("../shaders/4.1/depth_only.vert", "../shaders/4.1/empty.frag");
    auto debug_shadow_shader = iris::shader_t::create("../shaders/4.1/debug_shadow.vert", "../shaders/4.1/debug_shadow.frag");
    auto depth_reduce_init_shader = iris::shader_t::create_compute("../shaders/4.1/depth_reduce_init.comp");
    auto depth_reduce_shader = iris::shader_t::create_compute("../shaders/4.1/depth_reduce.comp");
    auto setup_shadows_shader = iris::shader_t::create_compute("../shaders/4.1/setup_shadows.comp");

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
    // light_positions.emplace_back( 0.0f, 0.5f,  0.0f);
    // light_positions.emplace_back( 3.0f, 0.5f,  0.0f);
    // light_positions.emplace_back( 3.0f, 0.5f,  3.0f);
    // light_positions.emplace_back( 3.0f, 0.5f, -3.0f);
    // light_positions.emplace_back(-3.0f, 2.5f,  3.0f);
    // light_positions.emplace_back(-3.0f, 2.5f, -3.0f);
    // light_positions.emplace_back( 3.0f, 2.5f,  3.0f);
    // light_positions.emplace_back( 6.0f, 0.5f,  3.0f);
    // light_positions.emplace_back( 6.0f, 0.5f, -3.0f);
    // light_positions.emplace_back(-6.0f, 0.5f,  3.0f);
    // light_positions.emplace_back(-6.0f, 0.5f, -3.0f);

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

    auto dir_light_sun = directional_light_t{
        .direction = glm::vec3(-2.25f, 35.0f, -6.5f),
        .diffuse = glm::vec3(0.8f),
        .specular = glm::vec3(0.5f),
    };

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

    // sRGB framebuffer
    glEnable(GL_FRAMEBUFFER_SRGB);

    // blending setup
    glDisable(GL_BLEND);
    // output.rgb = (input.rgb * input.a) + (previous.rgb * (1 - input.a))
    // output.a = (input.a * 1) + (previous.a * 0)
    // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
    // glBlendEquation(GL_FUNC_ADD);

    // face culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // multisampling
    // glEnable(GL_SAMPLE_ALPHA_TO_ONE);
    // glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);

    // mouse picking
    auto f0_main_attachments = std::to_array({
        iris::framebuffer_attachment_t::create(
            window.width,
            window.height,
            1,
            GL_SRGB8_ALPHA8,
            GL_RGBA,
            GL_UNSIGNED_BYTE),
        iris::framebuffer_attachment_t::create(
            window.width,
            window.height,
            1,
            GL_R32UI,
            GL_RED_INTEGER,
            GL_UNSIGNED_INT),
        iris::framebuffer_attachment_t::create(
            window.width,
            window.height,
            1,
            GL_DEPTH24_STENCIL8,
            GL_DEPTH_STENCIL,
            GL_UNSIGNED_INT_24_8),
    });

    auto f1_shadow_attachments = std::to_array({
        iris::framebuffer_attachment_t::create(
            4096,
            4096,
            4,
            GL_DEPTH_COMPONENT32F,
            GL_DEPTH_COMPONENT,
            GL_FLOAT),
    });
    f1_shadow_attachments[0].bind();
    glTexParameteri(f1_shadow_attachments[0].target(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(f1_shadow_attachments[0].target(), GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

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

    auto f1_shadow = iris::framebuffer_t::create({
        std::cref(f1_shadow_attachments[0]),
    });
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

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
    auto shadow_camera_buffer = iris::buffer_t::create(sizeof(cascade_data_t[4]), GL_SHADER_STORAGE_BUFFER);
    auto model_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto point_light_buffer = iris::buffer_t::create(sizeof(point_light_t[32]), GL_SHADER_STORAGE_BUFFER);
    auto dir_light_buffer = iris::buffer_t::create(sizeof(directional_light_t[32]), GL_UNIFORM_BUFFER);
    auto cascade_setup_buffer = iris::buffer_t::create(sizeof(cascade_setup_data_t), GL_UNIFORM_BUFFER);
    auto cascade_out_buffer = iris::buffer_t::create(sizeof(cascade_data_t[4]), GL_SHADER_STORAGE_BUFFER);

    // depth reduce output
    // yes I know this isn't a framebuffer attachment
    auto depth_reduce_wgc = calculate_workgroup_count_from_wh(window.width, window.height);
    auto depth_reduce_outs = std::vector<iris::framebuffer_attachment_t>();
    depth_reduce_outs.reserve(32);
    for (const auto& each : depth_reduce_wgc) {
        depth_reduce_outs.emplace_back(iris::framebuffer_attachment_t::create(
            each.x,
            each.y,
            1,
            GL_RG32F,
            GL_RG,
            GL_FLOAT));
    }

    // raycasting
    auto hit_mesh = std::optional<std::pair<std::reference_wrapper<const iris::mesh_t>, iris::uint32>>();

    // avoid clipping objects behind the frustum
    glEnable(GL_DEPTH_CLAMP);

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
                    1,
                    GL_SRGB8_ALPHA8,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE),
                iris::framebuffer_attachment_t::create(
                    window.width,
                    window.height,
                    1,
                    GL_R32UI,
                    GL_RED_INTEGER,
                    GL_UNSIGNED_INT),
                iris::framebuffer_attachment_t::create(
                    window.width,
                    window.height,
                    1,
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

            depth_reduce_wgc = calculate_workgroup_count_from_wh(window.width, window.height);
            depth_reduce_outs.clear();
            for (const auto& each : depth_reduce_wgc) {
                depth_reduce_outs.emplace_back(iris::framebuffer_attachment_t::create(
                    each.x,
                    each.y,
                    1,
                    GL_RG32F,
                    GL_RG,
                    GL_FLOAT));
            }

            window.is_resized = false;
        }

        auto camera_data = camera_data_t {
            camera.projection(),
            camera.view(),
            camera.position()
        };

        dir_light_sun.direction = glm::vec3(-2.25f * glm::sin(current_time), 35.0f, -6.5f * glm::cos(current_time));

        // shadow camera
        const auto [shadow_frustums, global_shadow_pv] = calculate_shadow_frustum(camera, glm::normalize(dir_light_sun.direction));

        const auto cascade_setup = cascade_setup_data_t {
            global_shadow_pv,
            glm::inverse(camera.projection() * camera.view()),
            glm::make_vec4(camera.right()),
            glm::make_vec4(glm::normalize(dir_light_sun.direction)),
            camera.near(),
            camera.far(),
            static_cast<iris::float32>(f1_shadow_attachments[0].width())
        };

        camera_buffer.write(&camera_data, iris::size_bytes(camera_data));
        shadow_camera_buffer.write(shadow_frustums.data(), iris::size_bytes(shadow_frustums));
        model_buffer.write(transforms.data(), iris::size_bytes(transforms));
        point_light_buffer.write(point_lights.data(), iris::size_bytes(point_lights));
        dir_light_buffer.write(&dir_light_sun, iris::size_bytes(dir_light_sun));
        cascade_setup_buffer.write(&cascade_setup, iris::size_bytes(cascade_setup));

        f0_main.bind();
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glCullFace(GL_BACK);
        glViewport(0, 0, f0_main.width(), f0_main.height());
        glScissor(0, 0, f0_main.width(), f0_main.height());
        const auto clear_depth = glm::vec4(1.0f);
        glClearBufferfv(GL_DEPTH, 0, glm::value_ptr(clear_depth));
        for (const auto& [ref, id] : scene.meshes) {
            const auto& mesh = ref.get();
            depth_only_shader
                .bind()
                .set(0, { id });
            camera_buffer.bind_base(0);
            model_buffer.bind_base(1);
            mesh.draw();
        }

        // depth reduce
        glActiveTexture(GL_TEXTURE0);
        f0_main_attachments[2].bind();
        depth_reduce_init_shader
            .bind()
            .set(0, { 0_i32 })
            .set(1, { camera.near() })
            .set(2, { camera.far() });
        glBindImageTexture(0, depth_reduce_outs[0].id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
        camera_buffer.bind_base(1);
        glDispatchCompute(depth_reduce_wgc[0].x, depth_reduce_wgc[0].y, 1);

        depth_reduce_shader.bind();
        for (auto i = 1_u32; i < depth_reduce_outs.size(); ++i) {
            glBindImageTexture(0, depth_reduce_outs[i - 1].id(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
            glBindImageTexture(1, depth_reduce_outs[i].id(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
            glDispatchCompute(depth_reduce_wgc[i].x, depth_reduce_wgc[i].y, 1);
        }

        // setup shadow cascades
        setup_shadows_shader.bind();
        f0_main_attachments[2].bind();
        glBindImageTexture(0, depth_reduce_outs.back().id(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
        cascade_setup_buffer.bind_base(1);
        cascade_out_buffer.bind_base(2);
        glDispatchCompute(1, 1, 1);

        {
            const auto draw_attachments = std::to_array<GLenum>({
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1
            });
            glDrawBuffers(2, draw_attachments.data());
            glReadBuffer(GL_COLOR_ATTACHMENT1);
        }
        // render the shadow map
        f1_shadow.bind();
        glCullFace(GL_FRONT);
        for (auto i = 0_u32; i < shadow_frustums.size(); ++i) {
            f1_shadow.set_layer(0, i);
            glViewport(0, 0, f1_shadow.width(), f1_shadow.height());
            glScissor(0, 0, f1_shadow.width(), f1_shadow.height());
            glClearDepth(1.0f);
            glClear(GL_DEPTH_BUFFER_BIT);
            for (const auto& [ref, id] : scene.meshes) {
                const auto& mesh = ref.get();
                shadow_shader
                    .bind()
                    .set(0, { id })
                    .set(1, { i });
                cascade_out_buffer.bind_base(0);
                //shadow_camera_buffer.bind_base(0);
                model_buffer.bind_base(1);
                mesh.draw();
            }
        }

        // 1. render the frustum lines
        f0_main.bind();
        glScissor(0, 0, window.width, window.height);
        glViewport(0, 0, window.width, window.height);
        const auto clear_color = glm::vec4(0.05f, 0.05f, 0.05f, 1.0f);
        const auto clear_id = glm::uvec4(0xffffffff);
        glClearBufferfv(GL_COLOR, 0, glm::value_ptr(clear_color));
        glClearBufferuiv(GL_COLOR, 1, glm::value_ptr(clear_id));
        glDepthFunc(GL_LEQUAL);
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
        glCullFace(GL_BACK);
        // 2. render opaque objects
        for (const auto& mesh : scene.opaque_meshes) {
            const auto& u_mesh = mesh.ref.get();

            simple_shader
                .bind()
                .set(0, { mesh.id })
                .set(1, { (iris::uint32)shadow_frustums.size() });

            camera_buffer.bind_base(0);
            model_buffer.bind_range(1, 0, iris::size_bytes(transforms));
            cascade_out_buffer.bind_base(2);
            //shadow_camera_buffer.bind_base(2);
            point_light_buffer.bind_range(3, 0, iris::size_bytes(point_lights) + 1);
            dir_light_buffer.bind_base(4);

            for (auto j = 0_i32; const auto& texture : u_mesh.textures()) {
                texture.get().bind(j);
                simple_shader.set(4 + j, { j });
                j++;
            }
            simple_shader.set(6, { 32_u32 });
            simple_shader.set(7, { static_cast<iris::uint32>(point_lights.size()) });

            glActiveTexture(GL_TEXTURE2);
            f1_shadow_attachments[0].bind();
            simple_shader.set(8, { 2_i32 });

            u_mesh.draw();
        }

        // sort and draw transparent objects
        {
            auto indices = std::vector<iris::uint64>(scene.transparent_meshes.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&camera, &transforms, &scene](const auto& i, const auto& j) {
                const auto& a_mesh = scene.transparent_meshes[i].ref.get();
                const auto& b_mesh = scene.transparent_meshes[j].ref.get();
                const auto& a_aabb = a_mesh.aabb();
                const auto& b_aabb = b_mesh.aabb();

                const auto opaque_offset = scene.opaque_meshes.size();
                const auto a_center = transforms[opaque_offset + i][0] * glm::vec4(a_aabb.center, 1.0f);
                const auto b_center = transforms[opaque_offset + j][0] * glm::vec4(b_aabb.center, 1.0f);
                const auto a_distance = glm::distance(camera.position(), glm::vec3(a_center));
                const auto b_distance = glm::distance(camera.position(), glm::vec3(b_center));
                return a_distance > b_distance;
            });
        }

        for (const auto& mesh : scene.transparent_meshes) {
            const auto& u_mesh = mesh.ref.get();

            simple_shader
                .bind()
                .set(0, { mesh.id })
                .set(1, { (iris::uint32)shadow_frustums.size() });

            camera_buffer.bind_base(0);
            model_buffer.bind_range(1, 0, iris::size_bytes(transforms));
            cascade_out_buffer.bind_base(2);
            //shadow_camera_buffer.bind_base(2);
            point_light_buffer.bind_range(3, 0, iris::size_bytes(point_lights) + 1);
            dir_light_buffer.bind_base(4);

            for (auto j = 0_i32; const auto& texture : u_mesh.textures()) {
                texture.get().bind(j);
                simple_shader.set(4 + j, { j });
                j++;
            }
            simple_shader.set(6, { 32_u32 });
            simple_shader.set(7, { static_cast<iris::uint32>(point_lights.size()) });

            glActiveTexture(GL_TEXTURE2);
            f1_shadow.attachment(0).bind();
            simple_shader.set(8, { 2_i32 });

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
                        std::cref(scene.meshes[mesh_id].ref.get()),
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
        glActiveTexture(GL_TEXTURE0);
        f0_main.attachment(0).bind();
        fullscreen_shader.set(0, { 0_i32 });
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // render debug shadow
        if (glfwGetKey(window.handle, GLFW_KEY_F2) == GLFW_PRESS) {
            fullscreen_shader.bind();
            glDisable(GL_DEPTH_TEST);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            const auto aspect =
                static_cast<iris::float32>(window.height) /
                static_cast<iris::float32>(window.width);
            const auto debug_render_width = 512;
            const auto debug_render_height = debug_render_width * aspect;
            glViewport(
                0,
                window.height - debug_render_height + 1,
                debug_render_width,
                debug_render_height);
            glScissor(
                0,
                window.height - debug_render_height + 1,
                debug_render_width,
                debug_render_height);
            glClear(GL_COLOR_BUFFER_BIT);

            debug_shadow_shader
                .bind()
                .set(0, { 0_i32 });
            glBindVertexArray(f_quad_vao);
            glActiveTexture(GL_TEXTURE0);
            f1_shadow_attachments[0].bind();
            fullscreen_shader.set(0, { 0_i32 });
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glEnable(GL_DEPTH_TEST);
        }

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        // update the window (only cursor position for now) and camera.
        window.update();
        camera.update(delta_time);
    }
    return 0;
}
