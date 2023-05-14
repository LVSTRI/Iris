#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <array>
#include <tuple>
#include <utility>
#include <functional>
#include <optional>

#include <texture.hpp>
#include <shader.hpp>
#include <camera.hpp>
#include <utilities.hpp>
#include <mesh_pool.hpp>
#include <model.hpp>
#include <framebuffer.hpp>
#include <buffer.hpp>
#include <allocator.hpp>

#include <debug_break.hpp>

#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#define CASCADE_COUNT 4
#define CULL_MODE_PERSPECTIVE_CAMERA 0
#define CULL_MODE_ORTHOGRAPHIC_CAMERA 1

using namespace iris::literals;

constexpr auto WINDOW_WIDTH = 800;
constexpr auto WINDOW_HEIGHT = 600;

struct draw_elements_indirect_t {
    iris::uint32 count = {};
    iris::uint32 instance_count = {};
    iris::uint32 first_index = {};
    iris::int32 base_vertex = {};
    iris::uint32 base_instance = {};
};

struct draw_arrays_indirect_t {
    iris::uint32 count = {};
    iris::uint32 instance_count = {};
    iris::uint32 first = {};
    iris::uint32 base_instance = {};
};

struct camera_data_t {
    glm::mat4 inf_projection;
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 pv;
    glm::vec3 position;
    iris::float32 near;
    iris::float32 far;
};

struct object_info_t {
    iris::uint32 local_transform = 0;
    iris::uint32 global_transform = 0;
    iris::uint32 diffuse_texture = 0;
    iris::uint32 normal_texture = 0;
    iris::uint32 specular_texture = 0;
    iris::uint32 group_index = 0;
    iris::uint32 group_offset = 0;
    iris::float32 _pad = 0;
    glm::vec4 scale = {};
    glm::vec4 sphere = {};
    iris::aabb_t aabb = {};
    draw_elements_indirect_t command = {};
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
    glm::mat4 global_pv = {};
    glm::mat4 inv_pv = {};
    glm::vec4 light_dir = {};
    iris::float32 resolution = {};
};

struct cascade_data_t {
    glm::mat4 inf_projection;
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 pv;
    glm::mat4 global;
    glm::vec4 scale;
    glm::vec4 offset; // w is split
};

struct cull_input_package_t {
    std::reference_wrapper<iris::buffer_t> indirect;
    std::reference_wrapper<iris::buffer_t> count;
    std::reference_wrapper<iris::buffer_t> shift;
};

static auto hash_combine(iris::uint64 seed, iris::uint64 value) noexcept -> iris::uint64 {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

struct indirect_group_t {
    std::vector<std::reference_wrapper<const iris::object_t>> objects;
    iris::uint32 vao = 0;
    iris::uint32 vbo = 0;
    iris::uint32 ebo = 0;
    iris::uint32 vertex_size = 0;
    iris::uint32 model_index = 0;
};

struct taa_pass_t {
    iris::framebuffer_attachment_t history;
    iris::framebuffer_attachment_t velocity;

    iris::framebuffer_attachment_t output;
    iris::framebuffer_t output_fbo;

    std::array<glm::vec2, 16> jitter;

    iris::uint32 frames = 0;
};

static auto group_indirect_commands(std::span<const iris::model_t> models) noexcept
    -> std::unordered_map<iris::uint64, indirect_group_t> {
    auto groups = std::unordered_map<iris::uint64, indirect_group_t>();
    auto model_index = 0_u32;
    for (const auto& model : models) {
        for (const auto& object : model.objects()) {
            auto hash = 0_u64;
            hash = hash_combine(hash, object.mesh.vao);
            hash = hash_combine(hash, object.mesh.vbo);
            hash = hash_combine(hash, object.mesh.ebo);
            hash = hash_combine(hash, object.mesh.vertex_slice.index());
            hash = hash_combine(hash, object.mesh.index_slice.index());
            groups[hash].objects.push_back(std::cref(object));
            groups[hash].vao = object.mesh.vao;
            groups[hash].vbo = object.mesh.vbo;
            groups[hash].ebo = object.mesh.ebo;
            groups[hash].vertex_size = object.mesh.vertex_size;
            groups[hash].model_index = model_index;
        }
        model_index++;
    }
    return groups;
}

static auto calculate_global_projection(const iris::camera_t& camera, const glm::vec3 light_dir) noexcept -> glm::mat4 {
    const auto ndc_cube = std::to_array({
        glm::vec3(-1.0f, -1.0f, 0.0f),
        glm::vec3( 1.0f, -1.0f, 0.0f),
        glm::vec3(-1.0f,  1.0f, 0.0f),
        glm::vec3( 1.0f,  1.0f, 0.0f),
        glm::vec3(-1.0f, -1.0f, 1.0f),
        glm::vec3( 1.0f, -1.0f, 1.0f),
        glm::vec3(-1.0f,  1.0f, 1.0f),
        glm::vec3( 1.0f,  1.0f, 1.0f),
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
    const auto projection = glm::ortho(min_ext.x, max_ext.x, min_ext.y, max_ext.y, 0.0f, 1.0f);
    const auto view = glm::lookAt(frustum_center + light_dir * 0.5f, frustum_center, glm::vec3(0.0f, 1.0f, 0.0f));
    const auto pv = projection * view;
    const auto uv_scale_bias = glm::mat4(
        glm::vec4(0.5f, 0.0f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
        glm::vec4(0.5f, 0.5f, 0.0f, 1.0f));
    return uv_scale_bias * pv;
}

static auto calculate_wg_from_resolution(iris::uint32 width, iris::uint32 height) noexcept -> std::vector<glm::uvec2> {
    constexpr auto wg_size = 16_u32;
    auto wg_count = std::vector<glm::uvec2>();
    wg_count.reserve(16);
    wg_count.emplace_back(
        (width + wg_size - 1) / wg_size,
        (height + wg_size - 1) / wg_size);
    while (wg_count.back() != glm::uvec2(1)) {
        const auto& previous = wg_count.back();
        const auto current_size = glm::uvec2(
            (previous.x + wg_size - 1) / wg_size,
            (previous.y + wg_size - 1) / wg_size);
        wg_count.emplace_back(glm::max(current_size, glm::uvec2(1)));
    }
    return wg_count;
}

static auto calculate_taa_jitter(glm::vec2 resolution) noexcept -> std::array<glm::vec2, 16> {
    return std::to_array({
        ((glm::vec2(0.500000f, 0.333333f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.250000f, 0.666667f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.750000f, 0.111111f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.125000f, 0.444444f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.625000f, 0.777778f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.375000f, 0.222222f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.875000f, 0.555556f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.062500f, 0.888889f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.562500f, 0.037037f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.312500f, 0.370370f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.812500f, 0.703704f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.187500f, 0.148148f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.687500f, 0.481481f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.437500f, 0.814815f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.937500f, 0.259259f) - 0.5f) / resolution) * 2.0f,
        ((glm::vec2(0.031250f, 0.592593f) - 0.5f) / resolution) * 2.0f,
    });
}

int main() {
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    iris_defer([]() {
        glfwTerminate();
    });

    auto window = iris::window_t();
    window.width = WINDOW_WIDTH;
    window.height = WINDOW_HEIGHT;
    window.handle = glfwCreateWindow(window.width, window.height, "Iris", nullptr, nullptr);
    if (!window.handle) {
        return -1;
    }

    glfwSetWindowUserPointer(window.handle, &window);
    glfwMakeContextCurrent(window.handle);

    if (!gladLoadGL(glfwGetProcAddress)) {
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

    glViewport(0, 0, window.width, window.height);
    glfwFocusWindow(window.handle);

    glfwSetFramebufferSizeCallback(window.handle, [](GLFWwindow* handle, int width, int height) {
        auto& window = *static_cast<iris::window_t*>(glfwGetWindowUserPointer(handle));
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

    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_REPRESENTATIVE_FRAGMENT_TEST_NV);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    auto main_shader = iris::shader_t::create("../shaders/5.2/main.vert", "../shaders/5.2/main.frag");
    auto depth_only_shader = iris::shader_t::create("../shaders/5.2/depth_only.vert", "../shaders/5.2/depth_only.frag");
    auto depth_reduce_init_shader = iris::shader_t::create_compute("../shaders/5.2/depth_reduce_init.comp");
    auto depth_reduce_shader = iris::shader_t::create_compute("../shaders/5.2/depth_reduce.comp");
    auto setup_cascades_shader = iris::shader_t::create_compute("../shaders/5.2/setup_shadows.comp");
    auto shadow_shader = iris::shader_t::create("../shaders/5.2/shadow.vert", "../shaders/5.2/shadow.frag");
    auto fullscreen_shader = iris::shader_t::create("../shaders/5.2/fullscreen.vert", "../shaders/5.2/fullscreen.frag");
    auto cull_shader = iris::shader_t::create_compute("../shaders/5.2/generic_cull.comp");
    auto roc_shader = iris::shader_t::create("../shaders/5.2/roc.vert", "../shaders/5.2/roc.frag");
    auto roc_cull_shader = iris::shader_t::create_compute("../shaders/5.2/roc_cull.comp");
    auto taa_resolve_shader = iris::shader_t::create("../shaders/5.2/taa_resolve.vert", "../shaders/5.2/taa_resolve.frag");

    // DEBUG
    auto debug_aabb_shader = iris::shader_t::create("../shaders/5.2/debug_aabb.vert", "../shaders/5.2/debug_aabb.frag");

    auto blue_noise_texture = iris::texture_t::create("../textures/1024_1024/LDR_RGBA_0.png", iris::texture_type_t::linear_r8g8b8_unorm);

    auto camera = iris::camera_t::create(window);

    auto mesh_pool = iris::mesh_pool_t::create();
    auto models = std::vector<iris::model_t>();
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/compressed/power_plant/power_plant.glb"));
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/compressed/sponza/sponza.glb"));
    models.emplace_back(iris::model_t::create(mesh_pool, "../models/compressed/bistro/bistro.glb"));
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/compressed/san_miguel/san_miguel.glb"));
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/compressed/cube/cube.glb"));
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/compressed/deccer_cubes/deccer_cubes.glb"));

    auto local_transforms = std::vector<glm::mat4>();
    local_transforms.reserve(16384);
    for (const auto& model : models) {
        std::ranges::copy(model.transforms(), std::back_inserter(local_transforms));
    }

    auto global_transforms = std::vector<glm::mat4>();
    global_transforms.emplace_back(glm::identity<glm::mat4>());

    auto objects = std::vector<std::reference_wrapper<const iris::object_t>>();
    objects.reserve(16384);
    for (const auto& model : models) {
        std::ranges::for_each(model.objects(), [&objects](const auto& mesh) {
            objects.emplace_back(std::cref(mesh));
        });
    }

    auto directional_lights = std::vector<directional_light_t>();
    directional_lights.push_back({
        .direction = glm::normalize(glm::vec3(-0.275f, 1.0f, 0.195f)),
        .diffuse = glm::vec3(1.0f, 1.0f, 1.0f),
        .specular = glm::vec3(1.0f, 1.0f, 1.0f)
    });

    auto empty_vao = 0_u32;
    glCreateVertexArrays(1, &empty_vao);

    auto aabb_vao = 0_u32;
    auto aabb_vbo = 0_u32;

    glCreateVertexArrays(1, &aabb_vao);
    glCreateBuffers(1, &aabb_vbo);
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
        glNamedBufferStorage(aabb_vbo, sizeof(aabb_vertices), aabb_vertices.data(), 0);
        glEnableVertexArrayAttrib(aabb_vao, 0);
        glVertexArrayAttribFormat(aabb_vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(aabb_vao, 0, 0);
        glVertexArrayVertexBuffer(aabb_vao, 0, aabb_vbo, 0, sizeof(glm::vec3));
    }

    auto camera_buffer = iris::buffer_t::create(sizeof(camera_data_t), GL_UNIFORM_BUFFER);
    auto frustum_buffer = iris::buffer_t::create(sizeof(iris::frustum_t[32]), GL_SHADER_STORAGE_BUFFER);
    auto local_transform_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto global_transform_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto object_info_buffer = iris::buffer_t::create(sizeof(object_info_t[16384]), GL_SHADER_STORAGE_BUFFER);
    auto texture_buffer = iris::buffer_t::create(sizeof(iris::uint64[4096]), GL_SHADER_STORAGE_BUFFER);
    auto cascade_setup_buffer = iris::buffer_t::create(sizeof(cascade_setup_data_t), GL_UNIFORM_BUFFER);
    auto directional_lights_buffer = iris::buffer_t::create(sizeof(directional_light_t[16]), GL_UNIFORM_BUFFER);
    auto cascade_buffer = iris::buffer_t::create(sizeof(cascade_data_t[CASCADE_COUNT]), GL_SHADER_STORAGE_BUFFER, GL_NONE);

    auto prev_camera_buffer = iris::buffer_t::create(sizeof(camera_data_t), GL_UNIFORM_BUFFER);
    auto prev_local_transform_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto prev_global_transform_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);

    // cull output
    auto main_indirect_buffer = iris::buffer_t::create(sizeof(draw_elements_indirect_t[16384]), GL_DRAW_INDIRECT_BUFFER);
    auto main_count_buffer = iris::buffer_t::create(sizeof(iris::uint64[1024]), GL_PARAMETER_BUFFER, GL_DYNAMIC_STORAGE_BIT);
    auto main_object_shift_buffer = iris::buffer_t::create(sizeof(iris::uint64[16384]), GL_SHADER_STORAGE_BUFFER, GL_NONE);

    auto shadow_indirect_buffer = iris::buffer_t::create(sizeof(draw_elements_indirect_t[16384]), GL_DRAW_INDIRECT_BUFFER, GL_NONE);
    auto shadow_count_buffer = iris::buffer_t::create(sizeof(iris::uint64[1024]), GL_PARAMETER_BUFFER, GL_NONE);
    auto shadow_object_shift_buffer = iris::buffer_t::create(sizeof(iris::uint64[16384]), GL_SHADER_STORAGE_BUFFER, GL_NONE);

    auto roc_indirect_buffer = iris::buffer_t::create(sizeof(draw_arrays_indirect_t), GL_DRAW_INDIRECT_BUFFER, GL_DYNAMIC_STORAGE_BIT);
    auto roc_object_shift_buffer = iris::buffer_t::create(sizeof(iris::uint64[16384]), GL_SHADER_STORAGE_BUFFER, GL_NONE);
    auto roc_visibility_buffer = iris::buffer_t::create(sizeof(iris::uint64[16384]), GL_SHADER_STORAGE_BUFFER, GL_NONE);

    // DEBUG
    auto debug_aabb_indirect_buffer = iris::buffer_t::create(sizeof(draw_arrays_indirect_t), GL_DRAW_INDIRECT_BUFFER, GL_DYNAMIC_STORAGE_BIT);

    auto taa_pass = taa_pass_t();
    taa_pass.history = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_SRGB8_ALPHA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        false,
        false);
    taa_pass.velocity = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_RG16F,
        GL_RG,
        GL_FLOAT,
        true,
        false);
    taa_pass.output = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_SRGB8_ALPHA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        false);
    taa_pass.output_fbo = iris::framebuffer_t::create({
        std::cref(taa_pass.output)
    });
    taa_pass.jitter = calculate_taa_jitter(glm::vec2(window.width, window.height));

    auto offscreen_attachment = std::vector<iris::framebuffer_attachment_t>(2);
    offscreen_attachment[0] = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_SRGB8_ALPHA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        true,
        true);
    offscreen_attachment[1] = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_DEPTH_COMPONENT32F,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        true,
        false);

    auto shadow_attachment = iris::framebuffer_attachment_t::create(
        4096,
        4096,
        CASCADE_COUNT,
        GL_DEPTH_COMPONENT16,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        false);
    glTextureParameteri(shadow_attachment.id(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(shadow_attachment.id(), GL_TEXTURE_COMPARE_FUNC, GL_LESS);

    auto depth_reduce_wgc = calculate_wg_from_resolution(window.width, window.height);
    auto depth_reduce_attachments = std::vector<iris::framebuffer_attachment_t>();
    depth_reduce_attachments.reserve(depth_reduce_wgc.size());
    for (const auto& each : depth_reduce_wgc) {
        depth_reduce_attachments.emplace_back(iris::framebuffer_attachment_t::create(
            each.x,
            each.y,
            1,
            GL_RG32F,
            GL_RG,
            GL_FLOAT));
    }

    auto offscreen_fbo = iris::framebuffer_t::create({
        std::cref(offscreen_attachment[0]),
        std::cref(taa_pass.velocity),
        std::cref(offscreen_attachment[1])
    });
    glNamedFramebufferDrawBuffers(offscreen_fbo.id(), 2, std::to_array<iris::uint32>({
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_ATTACHMENT1
    }).data());

    auto depth_only_fbo = iris::framebuffer_t::create({
        std::cref(offscreen_attachment[1])
    });
    depth_only_fbo.clear_depth(1.0f);

    auto shadow_fbo = iris::framebuffer_t::create({
        std::cref(shadow_attachment)
    });

    auto delta_time = 0.0f;
    auto last_time = 0.0f;
    glfwSwapInterval(0);
    auto last_key_c = false;
    auto freeze_frustum_culling = false;
    auto prev_camera_data = camera_data_t {
        camera.projection(true),
        camera.projection(),
        camera.view(),
        camera.projection() * camera.view(),
        camera.position(),
        camera.near(),
        camera.far()
    };
    auto prev_global_transforms = global_transforms;
    auto prev_local_transforms = local_transforms;
    while (!glfwWindowShouldClose(window.handle)) {
        if (window.is_resized) {
            taa_pass.jitter = calculate_taa_jitter(glm::vec2(window.width, window.height));
            taa_pass.history = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_SRGB8_ALPHA8,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                false,
                false);
            taa_pass.velocity = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_RG16F,
                GL_RG,
                GL_FLOAT,
                true,
                false);
            taa_pass.output = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_SRGB8_ALPHA8,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                false);
            taa_pass.output_fbo = iris::framebuffer_t::create({
                std::cref(taa_pass.output)
            });
            taa_pass.frames = -2;

            offscreen_attachment[0] = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_SRGB8_ALPHA8,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                false,
                true);
            offscreen_attachment[1] = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_DEPTH_COMPONENT32F,
                GL_DEPTH_COMPONENT,
                GL_FLOAT,
                true,
                false);
            offscreen_fbo = iris::framebuffer_t::create({
                std::cref(offscreen_attachment[0]),
                std::cref(taa_pass.velocity),
                std::cref(offscreen_attachment[1])
            });
            glNamedFramebufferDrawBuffers(offscreen_fbo.id(), 2, std::to_array<iris::uint32>({
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1
            }).data());

            depth_only_fbo = iris::framebuffer_t::create({
                std::cref(offscreen_attachment[1])
            });

            depth_reduce_wgc = calculate_wg_from_resolution(window.width, window.height);
            depth_reduce_attachments.clear();
            for (const auto& each : depth_reduce_wgc) {
                depth_reduce_attachments.emplace_back(iris::framebuffer_attachment_t::create(
                    each.x,
                    each.y,
                    1,
                    GL_RG32F,
                    GL_RG,
                    GL_FLOAT));
            }
            window.is_resized = false;
        }
        if (glfwGetKey(window.handle, GLFW_KEY_H) == GLFW_PRESS) {
            taa_pass.frames = 0;
        }

        const auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_time;
        last_time = current_time;

        const auto key_c_pressed = glfwGetKey(window.handle, GLFW_KEY_C) == GLFW_PRESS;
        if (key_c_pressed && !last_key_c) {
            freeze_frustum_culling = !freeze_frustum_culling;
        }
        last_key_c = key_c_pressed;

        /*directional_lights[0].direction = glm::normalize(glm::vec3(
            0.33f * glm::cos(current_time * 0.5f),
            1.0f,
            0.33f * glm::sin(current_time * 0.5f)));*/

        auto object_infos = std::vector<object_info_t>();
        object_infos.reserve(objects.size());
        const auto indirect_groups = group_indirect_commands(models);
        {
            auto mesh_index = 0_u32;
            auto group_index = 0_u32;
            auto group_offset = 0_u32;
            for (auto& [_, group] : indirect_groups) {
                for (auto& object : group.objects) {
                    auto texture_offset = 0_u32;
                    for (auto i = 0_u32; i < group.model_index; ++i) {
                        texture_offset += models[i].textures().size();
                    }
                    auto& u_object = object.get();
                    auto command = draw_elements_indirect_t {
                        static_cast<iris::uint32>(u_object.mesh.index_count),
                        1,
                        static_cast<iris::uint32>(u_object.mesh.index_offset),
                        static_cast<iris::int32>(u_object.mesh.vertex_offset),
                        0
                    };
                    object_infos.push_back({
                        .local_transform = mesh_index,
                        .global_transform = group.model_index,
                        .diffuse_texture = u_object.diffuse_texture + texture_offset,
                        .normal_texture = u_object.normal_texture + texture_offset,
                        .specular_texture = u_object.specular_texture + texture_offset,
                        .group_index = group_index,
                        .group_offset = group_offset,
                        .scale = glm::make_vec4(u_object.scale),
                        .sphere = u_object.sphere,
                        .aabb = u_object.aabb,
                        .command = command
                    });
                    mesh_index++;
                }
                group_offset += group.objects.size();
                group_index++;
            }
        }

        roc_indirect_buffer.write(iris::as_const_ptr(draw_arrays_indirect_t {
            .count = 24,
            .instance_count = 0,
            .first = 0,
            .base_instance = 0
        }), sizeof(draw_arrays_indirect_t));

        auto texture_handles = std::vector<iris::uint64>();
        texture_handles.reserve(object_infos.size());
        for (const auto& model : models) {
            std::ranges::for_each(model.textures(), [&texture_handles](const auto& texture) {
                texture_handles.emplace_back(texture.handle());
            });
        }

        const auto camera_frustum = iris::make_perspective_frustum(camera.projection() * camera.view());
        const auto camera_data = camera_data_t {
            camera.projection(true),
            camera.projection(),
            camera.view(),
            camera.projection() * camera.view(),
            camera.position(),
            camera.near(),
            camera.far()
        };
        camera_buffer.write(iris::as_const_ptr(camera_data), sizeof(camera_data_t));
        prev_camera_buffer.write(iris::as_const_ptr(prev_camera_data), sizeof(camera_data_t));

        auto global_pv = calculate_global_projection(camera, directional_lights[0].direction);
        cascade_setup_buffer.write(iris::as_const_ptr(cascade_setup_data_t {
            .global_pv = global_pv,
            .inv_pv = glm::inverse(camera.projection() * camera.view()),
            .light_dir = glm::make_vec4(directional_lights[0].direction),
            .resolution = static_cast<iris::float32>(shadow_attachment.width()),
        }), sizeof(cascade_setup_data_t));

        local_transform_buffer.write(local_transforms.data(), iris::size_bytes(local_transforms));
        global_transform_buffer.write(global_transforms.data(), iris::size_bytes(global_transforms));
        object_info_buffer.write(object_infos.data(), iris::size_bytes(object_infos));
        texture_buffer.write(texture_handles.data(), iris::size_bytes(texture_handles));
        directional_lights_buffer.write(directional_lights.data(), iris::size_bytes(directional_lights));

        prev_local_transform_buffer.write(prev_local_transforms.data(), iris::size_bytes(prev_local_transforms));
        prev_global_transform_buffer.write(prev_global_transforms.data(), iris::size_bytes(prev_global_transforms));

        auto frustum_cull_scene = [&](
            cull_input_package_t package,
            iris::uint32 disable_near,
            iris::uint32 cascade_layer = -1) {
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "frustum_cull_pass");
            cull_shader
                .bind()
                .set(0, { static_cast<iris::uint32>(indirect_groups.size()) })
                .set(1, { static_cast<iris::uint32>(object_infos.size()) })
                .set(2, { 0_u32 })
                .set(3, { disable_near })
                .set(4, { cascade_layer });
            local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
            global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
            object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
            package.indirect.get().bind_base(GL_SHADER_STORAGE_BUFFER, 4);
            package.count.get().bind_base(GL_SHADER_STORAGE_BUFFER, 5);
            package.shift.get().bind_base(6);
            cascade_buffer.bind_base(7);
            camera_buffer.bind_base(8);
            roc_indirect_buffer.bind_base(GL_SHADER_STORAGE_BUFFER, 9);
            roc_object_shift_buffer.bind_base(10);

            glClearNamedBufferSubData(
                package.count.get().id(),
                GL_R32UI,
                0,
                package.count.get().size(),
                GL_RED_INTEGER,
                GL_UNSIGNED_INT,
                nullptr);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            glDispatchCompute((object_infos.size() + 255) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BUFFER | GL_COMMAND_BARRIER_BIT);
            glPopDebugGroup();
        };

        auto main_indirect_package = cull_input_package_t {
            .indirect = std::ref(main_indirect_buffer),
            .count = std::ref(main_count_buffer),
            .shift = std::ref(main_object_shift_buffer)
        };
        auto shadow_indirect_package = cull_input_package_t {
            .indirect = std::ref(shadow_indirect_buffer),
            .count = std::ref(shadow_count_buffer),
            .shift = std::ref(shadow_object_shift_buffer)
        };

        glViewport(0, 0, window.width, window.height);

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "depth_prepass");
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        frustum_buffer.write(iris::as_const_ptr(camera_frustum), iris::size_bytes(camera_frustum));
        frustum_buffer.bind_range(0, 0, iris::size_bytes(camera_frustum));
        if (!freeze_frustum_culling) {
            frustum_cull_scene(main_indirect_package, 0);
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "main_roc");
            glDisable(GL_CULL_FACE);
            glDepthMask(GL_FALSE);
            glClearNamedBufferSubData(
                roc_visibility_buffer.id(),
                GL_R32UI,
                0,
                roc_visibility_buffer.size(),
                GL_RED_INTEGER,
                GL_UNSIGNED_INT,
                nullptr);
            depth_only_fbo.bind();
            roc_shader.bind();
            camera_buffer.bind_base(0);
            local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
            global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
            object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
            roc_object_shift_buffer.bind_base(4);
            roc_visibility_buffer.bind_base(5);
            roc_indirect_buffer.bind();
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            glBindVertexArray(empty_vao);
            glMultiDrawArraysIndirect(GL_TRIANGLE_STRIP, nullptr, 1, 0);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
            glDepthMask(GL_TRUE);
            glEnable(GL_CULL_FACE);
            roc_cull_shader
                .bind()
                .set(0, { static_cast<iris::uint32>(objects.size()) });
            object_info_buffer.bind_range(0, 0, iris::size_bytes(object_infos));
            roc_visibility_buffer.bind_base(1);
            main_indirect_buffer.bind_base(GL_SHADER_STORAGE_BUFFER, 2);
            main_count_buffer.bind_base(GL_SHADER_STORAGE_BUFFER, 3);
            main_object_shift_buffer.bind_base(4);
            glClearNamedBufferSubData(
                main_count_buffer.id(),
                GL_R32UI,
                0,
                main_count_buffer.size(),
                GL_RED_INTEGER,
                GL_UNSIGNED_INT,
                nullptr);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            glDispatchCompute((objects.size() + 255) / 256, 1, 1);
            glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
            glPopDebugGroup();
        }

        depth_only_fbo.clear_depth(1.0f);
        depth_only_fbo.bind();
        depth_only_shader
            .bind()
            .set(1, taa_pass.jitter[taa_pass.frames % 16]);
        camera_buffer.bind_base(0);
        local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
        global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
        object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
        main_object_shift_buffer.bind_base(4);
        main_indirect_buffer.bind();
        main_count_buffer.bind();
        {
            auto indirect_offset = 0_u32;
            auto group_offset = 0_u32;
            auto group_count_offset = 0_u64;
            for (auto& [_, group] : indirect_groups) {
                depth_only_shader.set(0, { group_offset });
                glBindVertexArray(group.vao);
                glVertexArrayVertexBuffer(group.vao, 0, group.vbo, 0, group.vertex_size);
                glVertexArrayElementBuffer(group.vao, group.ebo);
                glMultiDrawElementsIndirectCount(
                    GL_TRIANGLES,
                    GL_UNSIGNED_INT,
                    reinterpret_cast<const void*>(indirect_offset),
                    static_cast<std::intptr_t>(group_count_offset),
                    static_cast<iris::int32>(group.objects.size()),
                    0);
                indirect_offset += group.objects.size() * sizeof(draw_elements_indirect_t);
                group_offset += group.objects.size();
                group_count_offset += sizeof(iris::uint32);
            }
        }
        glPopDebugGroup();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "depth_reduce");
        depth_reduce_init_shader.bind();
        offscreen_attachment[1].bind_texture(0);
        depth_reduce_attachments[0].bind_image_texture(0, 0, false, 0, GL_WRITE_ONLY);
        camera_buffer.bind_base(1);
        glDispatchCompute(depth_reduce_wgc[0].x, depth_reduce_wgc[0].y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        depth_reduce_shader.bind();
        for (auto i = 1; i < depth_reduce_wgc.size(); i++) {
            depth_reduce_attachments[i - 1].bind_image_texture(0, 0, false, 0, GL_READ_ONLY);
            depth_reduce_attachments[i].bind_image_texture(1, 0, false, 0, GL_WRITE_ONLY);
            glDispatchCompute(depth_reduce_wgc[i].x, depth_reduce_wgc[i].y, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        glPopDebugGroup();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "shadow_setup");
        setup_cascades_shader.bind();
        depth_reduce_attachments.back().bind_image_texture(0, 0, false, 0, GL_READ_ONLY);
        cascade_setup_buffer.bind_base(1);
        camera_buffer.bind_base(2);
        cascade_buffer.bind_base(3);
        frustum_buffer.bind_range(4, sizeof(iris::frustum_t), sizeof(iris::frustum_t[CASCADE_COUNT]));
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glPopDebugGroup();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "shadow_render");
        //glCullFace(GL_FRONT);
        glEnable(GL_DEPTH_CLAMP);
        for (auto layer = 0_u32; layer < CASCADE_COUNT; layer++) {
            // cull first
            shadow_fbo.bind();
            shadow_fbo.set_layer(0, layer);
            frustum_buffer.bind_range(0, (layer + 1) * sizeof(iris::frustum_t), sizeof(iris::frustum_t));
            // occlusion culling is disabled for shadows (for now)
            frustum_cull_scene(shadow_indirect_package, 1, layer);

            glViewport(0, 0, shadow_attachment.width(), shadow_attachment.height());
            shadow_shader
                .bind()
                .set(0, { layer });
            cascade_buffer.bind_base(0);
            local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
            global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
            object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
            shadow_object_shift_buffer.bind_base(4);
            texture_buffer.bind_range(5, 0, iris::size_bytes(texture_handles));
            shadow_count_buffer.bind();
            shadow_indirect_buffer.bind();

            shadow_fbo.clear_depth(1.0f);

            auto indirect_offset = 0_u32;
            auto group_offset = 0_u32;
            auto group_count_offset = 0_u64;
            for (auto& [_, group] : indirect_groups) {
                shadow_shader.set(1, { group_offset });
                glBindVertexArray(group.vao);
                glVertexArrayVertexBuffer(group.vao, 0, group.vbo, 0, group.vertex_size);
                glVertexArrayElementBuffer(group.vao, group.ebo);
                glMultiDrawElementsIndirectCount(
                    GL_TRIANGLES,
                    GL_UNSIGNED_INT,
                    reinterpret_cast<const void*>(indirect_offset),
                    static_cast<std::intptr_t>(group_count_offset),
                    static_cast<iris::int32>(group.objects.size()),
                    0);
                indirect_offset += group.objects.size() * sizeof(draw_elements_indirect_t);
                group_offset += group.objects.size();
                group_count_offset += sizeof(iris::uint32);
            }
        }
        glDisable(GL_DEPTH_CLAMP);
        //glCullFace(GL_BACK);
        glPopDebugGroup();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "final_color_pass");
        glViewport(0, 0, window.width, window.height);
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_EQUAL);

        offscreen_fbo.bind();
        offscreen_fbo.clear_color(0, { 0_u32, 0_u32, 0_u32, 255_u32 });
        offscreen_fbo.clear_color(1, { 0_u32, 0_u32, 0_u32, 255_u32 });
        frustum_buffer.bind_range(0, 0, iris::size_bytes(camera_frustum));

        main_shader
            .bind()
            .set(1, taa_pass.jitter[taa_pass.frames % 16])
            .set(4, glm::vec2(window.width, window.height));
        camera_buffer.bind_base(0);
        local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
        global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
        object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
        main_object_shift_buffer.bind_base(4);
        directional_lights_buffer.bind_range(5, 0, iris::size_bytes(directional_lights));
        texture_buffer.bind_range(6, 0, iris::size_bytes(texture_handles));
        cascade_buffer.bind_base(7);
        prev_camera_buffer.bind_base(8);
        prev_local_transform_buffer.bind_range(9, 0, iris::size_bytes(prev_local_transforms));
        prev_global_transform_buffer.bind_range(10, 0, iris::size_bytes(prev_global_transforms));
        shadow_attachment.bind_texture(0);
        blue_noise_texture.bind(1);
        main_indirect_buffer.bind();
        main_count_buffer.bind();
        {
            auto indirect_offset = 0_u32;
            auto group_offset = 0_u32;
            auto group_count_offset = 0_u64;
            main_shader
                .set(2, { 0_i32 })
                .set(3, { 1_i32 });
            for (auto& [_, group] : indirect_groups) {
                main_shader.set(0, { group_offset });
                glBindVertexArray(group.vao);
                glVertexArrayVertexBuffer(group.vao, 0, group.vbo, 0, group.vertex_size);
                glVertexArrayElementBuffer(group.vao, group.ebo);
                glMultiDrawElementsIndirectCount(
                    GL_TRIANGLES,
                    GL_UNSIGNED_INT,
                    reinterpret_cast<const void*>(indirect_offset),
                    static_cast<std::intptr_t>(group_count_offset),
                    static_cast<iris::int32>(group.objects.size()),
                    0);
                indirect_offset += group.objects.size() * sizeof(draw_elements_indirect_t);
                group_offset += group.objects.size();
                group_count_offset += sizeof(iris::uint32);
            }
        }
        glPopDebugGroup();

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "taa_resolve_pass");
        glDisable(GL_DEPTH_TEST);
        if (taa_pass.frames > 0) {
            taa_resolve_shader
                .bind()
                .set(0, { 0_i32 })
                .set(1, { 1_i32 })
                .set(2, { 2_i32 });
            offscreen_attachment[0].bind_texture(0);
            taa_pass.history.bind_texture(1);
            taa_pass.velocity.bind_texture(2);
            taa_pass.output_fbo.bind();
            glBindVertexArray(empty_vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glCopyImageSubData(
                taa_pass.output.id(),
                GL_TEXTURE_2D,
                0, 0, 0, 0,
                taa_pass.history.id(),
                GL_TEXTURE_2D,
                0, 0, 0, 0,
                window.width, window.height, 1);
        } else {
            glCopyImageSubData(
                offscreen_attachment[0].id(),
                GL_TEXTURE_2D,
                0, 0, 0, 0,
                taa_pass.history.id(),
                GL_TEXTURE_2D,
                0, 0, 0, 0,
                window.width, window.height, 1);
            glCopyImageSubData(
                offscreen_attachment[0].id(),
                GL_TEXTURE_2D,
                0, 0, 0, 0,
                taa_pass.output.id(),
                GL_TEXTURE_2D,
                0, 0, 0, 0,
                window.width, window.height, 1);
        }
        glPopDebugGroup();

        if (glfwGetKey(window.handle, GLFW_KEY_F) == GLFW_PRESS) {
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "debug_aabbs");
            glDisable(GL_CULL_FACE);
            debug_aabb_shader.bind();
            camera_buffer.bind_base(0);
            local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
            global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
            object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
            main_object_shift_buffer.bind_base(4);
            {
                auto group_offset = 0_u32;
                auto group_count_offset = 0_u64;
                debug_aabb_indirect_buffer.bind();
                for (auto& [_, group] : indirect_groups) {
                    auto command = draw_arrays_indirect_t {
                        .count = 24,
                        .instance_count = 0,
                        .first = 0,
                        .base_instance = group_offset
                    };
                    debug_aabb_indirect_buffer.write(&command, sizeof(command));
                    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
                    glCopyNamedBufferSubData(
                        main_count_buffer.id(),
                        debug_aabb_indirect_buffer.id(),
                        group_count_offset,
                        sizeof(iris::uint32),
                        sizeof(iris::uint32));
                    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
                    glBindVertexArray(aabb_vao);
                    glMultiDrawArraysIndirect(GL_LINES, nullptr, 1, 0);
                    group_offset += group.objects.size();
                    group_count_offset += sizeof(iris::uint32);
                }
            }
            glPopDebugGroup();
        }

        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "copy_to_backbuffer");
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        fullscreen_shader.bind();
        taa_pass.output.bind_texture(0);
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glPopDebugGroup();

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        window.update();
        camera.update(delta_time);
        prev_camera_data = camera_data;
        prev_global_transforms = global_transforms;
        prev_local_transforms = local_transforms;
        taa_pass.frames++;
    }
    return 0;
}