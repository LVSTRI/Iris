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
    const auto projection = glm::ortho(min_ext.x, max_ext.x, min_ext.y, max_ext.y, 0.0f, 1.0f);
    const auto view = glm::lookAt(frustum_center + light_dir * 0.5f, frustum_center, glm::vec3(0.0f, 1.0f, 0.0f));
    const auto pv = projection * view;
    const auto uv_scale_bias = glm::mat4(
        glm::vec4(0.5f, 0.0f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 0.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 0.0f),
        glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
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
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
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
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    auto camera = iris::camera_t::create(window);
    auto model = iris::meshlet_model_t::create("../models/compressed/bistro/bistro.glb");

    auto vertices = std::vector<iris::meshlet_vertex_format_t>();
    vertices.insert_range(vertices.end(), model.vertices());

    auto indices = std::vector<iris::uint32>();
    indices.insert_range(indices.end(), model.indices());

    auto triangles = std::vector<iris::uint32>();
    triangles.insert_range(triangles.end(), model.triangles());

    struct raw_meshlet_t {
        iris::meshlet_t meshlet = {};
        // custom data
        // index of the mesh this meshlet belongs to
        iris::uint32 mesh_index = 0;
    };
    auto meshlets = std::vector<raw_meshlet_t>();
    meshlets.reserve(model.meshlet_groups().size() * 128);
    {
        for (auto mesh_index = 0_u32; const auto& meshlet_group : model.meshlet_groups()) {
            for (const auto& meshlet : meshlet_group.meshlets) {
                meshlets.push_back({
                    meshlet,
                    mesh_index
                });
            }
            mesh_index++;
        }
    }

    auto object_info = std::vector<object_info_t>();
    object_info.reserve(model.meshlet_groups().size());
    {
        for (const auto& meshlet_group : model.meshlet_groups()) {
            object_info.push_back({
                .diffuse_texture = meshlet_group.diffuse_index,
                .normal_texture = meshlet_group.normal_index,
                .specular_texture = meshlet_group.specular_index,
            });
        }
    }

    auto texture_handles = std::vector<iris::uint64>();
    texture_handles.reserve(model.textures().size());
    for (const auto& texture : model.textures()) {
        texture_handles.push_back(texture.handle());
    }

    auto transforms = model.transforms();

    auto vertex_buffer = 0_u32;
    glCreateBuffers(1, &vertex_buffer);
    glNamedBufferStorage(vertex_buffer, iris::size_bytes(vertices), vertices.data(), GL_NONE);

    auto index_buffer = 0_u32;
    glCreateBuffers(1, &index_buffer);
    glNamedBufferStorage(index_buffer, iris::size_bytes(indices), indices.data(), GL_NONE);

    auto triangle_buffer = 0_u32;
    glCreateBuffers(1, &triangle_buffer);
    glNamedBufferStorage(triangle_buffer, iris::size_bytes(triangles), triangles.data(), GL_NONE);

    auto meshlet_buffer = 0_u32;
    glCreateBuffers(1, &meshlet_buffer);
    glNamedBufferStorage(meshlet_buffer, iris::size_bytes(meshlets), meshlets.data(), GL_NONE);

    auto transform_buffer = 0_u32;
    glCreateBuffers(1, &transform_buffer);
    glNamedBufferStorage(transform_buffer, iris::size_bytes(transforms), transforms.data(), GL_NONE);

    auto texture_buffer = 0_u32;
    glCreateBuffers(1, &texture_buffer);
    glNamedBufferStorage(texture_buffer, iris::size_bytes(texture_handles), texture_handles.data(), GL_NONE);

    auto object_buffer = 0_u32;
    glCreateBuffers(1, &object_buffer);
    glNamedBufferStorage(object_buffer, iris::size_bytes(object_info), object_info.data(), GL_NONE);

    auto main_shader = iris::shader_t::create_mesh("", "../shaders/5.1/main.mesh", "../shaders/5.1/main.frag");

    auto delta_time = 0.0f;
    auto last_time = 0.0f;
    glfwSwapInterval(0);
    while (!glfwWindowShouldClose(window.handle)) {
        if (window.is_resized) {
            window.is_resized = false;
        }

        auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_time;
        last_time = current_time;

        main_shader
            .bind()
            .set(0, { camera.projection() * camera.view() });
        glViewport(0, 0, window.width, window.height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClearDepth(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, transform_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, meshlet_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vertex_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, index_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, triangle_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, object_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, texture_buffer);
        glDrawMeshTasksNV(0, model.meshlet_count());

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        window.update();
        camera.update(delta_time);
    }
    return 0;
}