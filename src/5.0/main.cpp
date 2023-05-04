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

#include <stb_image.h>

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

static auto hash_combine(iris::uint64 seed, iris::uint64 value) noexcept -> iris::uint64 {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

static auto group_indirect_commands(std::span<std::reference_wrapper<const iris::mesh_t>> meshes) noexcept
    -> std::unordered_map<iris::uint64, std::vector<std::reference_wrapper<const iris::mesh_t>>> {
    auto groups = std::unordered_map<iris::uint64, std::vector<std::reference_wrapper<const iris::mesh_t>>>();
    for (const auto& mesh : meshes) {
        auto& u_mesh = mesh.get();
        auto hash = 0_u64;
        hash = hash_combine(hash, u_mesh.vao);
        hash = hash_combine(hash, u_mesh.vbo);
        hash = hash_combine(hash, u_mesh.ebo);
        hash = hash_combine(hash, u_mesh.vertex_slice.index());
        hash = hash_combine(hash, u_mesh.index_slice.index());
        groups[hash].push_back(mesh);
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
    glEnable(GL_DEPTH_CLAMP);

    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    auto main_shader = iris::shader_t::create("../shaders/5.0/main.vert", "../shaders/5.0/main.frag");
    auto depth_only_shader = iris::shader_t::create("../shaders/5.0/depth_only.vert", "../shaders/5.0/empty.frag");
    auto depth_reduce_init_shader = iris::shader_t::create_compute("../shaders/5.0/depth_reduce_init.comp");
    auto depth_reduce_shader = iris::shader_t::create_compute("../shaders/5.0/depth_reduce.comp");
    auto setup_cascades_shader = iris::shader_t::create_compute("../shaders/5.0/setup_shadows.comp");
    auto shadow_shader = iris::shader_t::create("../shaders/5.0/shadow.vert", "../shaders/5.0/shadow.frag");
    auto fullscreen_shader = iris::shader_t::create("../shaders/5.0/fullscreen.vert", "../shaders/5.0/fullscreen.frag");

    auto blue_noise_texture = iris::texture_t::create("../textures/1024_1024/LDR_RGBA_0.png", iris::texture_type_t::linear_srgb);

    auto camera = iris::camera_t::create(window);

    auto mesh_pool = iris::mesh_pool_t::create();
    auto models = std::vector<iris::model_t>();
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/sponza/Sponza.gltf"));
    models.emplace_back(iris::model_t::create(mesh_pool, "../models/bistro/bistro.gltf"));
    //models.emplace_back(iris::model_t::create(mesh_pool, "../models/san_miguel/san_miguel.gltf"));

    auto local_transforms = std::vector<glm::mat4>();
    for (const auto& model : models) {
        local_transforms.insert(local_transforms.end(), model.transforms().begin(), model.transforms().end());
    }

    auto global_transforms = std::vector<glm::mat4>();
    global_transforms.emplace_back(glm::identity<glm::mat4>());

    auto meshes = std::vector<std::reference_wrapper<const iris::mesh_t>>();
    meshes.reserve(16384);
    for (const auto& model : models) {
        std::ranges::for_each(model.meshes(), [&meshes](const auto& mesh) {
            meshes.emplace_back(std::cref(mesh));
        });
    }

    auto directional_lights = std::vector<directional_light_t>();
    directional_lights.push_back({
        .direction = glm::normalize(glm::vec3(0.33f, 1.0f, 0.5f)),
        .diffuse = glm::vec3(1.0f, 1.0f, 1.0f),
        .specular = glm::vec3(1.0f, 1.0f, 1.0f)
    });

    auto empty_vao = 0_u32;
    glCreateVertexArrays(1, &empty_vao);

    // indirect buffer
    auto ibo = iris::buffer_t::create(sizeof(draw_elements_indirect_t[16384]), GL_DRAW_INDIRECT_BUFFER);

    auto camera_buffer = iris::buffer_t::create(sizeof(camera_data_t), GL_UNIFORM_BUFFER);
    auto local_transform_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto global_transform_buffer = iris::buffer_t::create(sizeof(glm::mat4[16384]), GL_SHADER_STORAGE_BUFFER);
    auto object_info_buffer = iris::buffer_t::create(sizeof(object_info_t[16384]), GL_SHADER_STORAGE_BUFFER);
    auto texture_buffer = iris::buffer_t::create(sizeof(iris::uint64[16384]), GL_SHADER_STORAGE_BUFFER);
    auto cascade_setup_buffer = iris::buffer_t::create(sizeof(cascade_setup_data_t), GL_UNIFORM_BUFFER);
    auto cascade_buffer = iris::buffer_t::create(sizeof(cascade_data_t[CASCADE_COUNT]), GL_SHADER_STORAGE_BUFFER);
    auto directional_lights_buffer = iris::buffer_t::create(sizeof(directional_light_t[16]), GL_UNIFORM_BUFFER);

    auto offscreen_attachment = std::vector<iris::framebuffer_attachment_t>(2);
    offscreen_attachment[0] = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_SRGB8_ALPHA8,
        GL_RGBA,
        GL_UNSIGNED_BYTE);
    offscreen_attachment[1] = iris::framebuffer_attachment_t::create(
        window.width,
        window.height,
        1,
        GL_DEPTH_COMPONENT32F,
        GL_DEPTH_COMPONENT,
        GL_FLOAT);

    auto shadow_attachment = iris::framebuffer_attachment_t::create(
        4096,
        4096,
        CASCADE_COUNT,
        GL_DEPTH_COMPONENT32F,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        false);
    glTextureParameteri(shadow_attachment.id(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(shadow_attachment.id(), GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

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
        std::cref(offscreen_attachment[1])
    });

    auto depth_only_fbo = iris::framebuffer_t::create({
        std::cref(offscreen_attachment[1])
    });

    auto shadow_fbo = iris::framebuffer_t::create({
        std::cref(shadow_attachment)
    });

    auto delta_time = 0.0f;
    auto last_time = 0.0f;
    while (!glfwWindowShouldClose(window.handle)) {
        if (window.is_resized) {
            offscreen_attachment[0] = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_SRGB8_ALPHA8,
                GL_RGBA,
                GL_UNSIGNED_BYTE);
            offscreen_attachment[1] = iris::framebuffer_attachment_t::create(
                window.width,
                window.height,
                1,
                GL_DEPTH_COMPONENT32F,
                GL_DEPTH_COMPONENT,
                GL_FLOAT);

            offscreen_fbo = iris::framebuffer_t::create({
                std::cref(offscreen_attachment[0]),
                std::cref(offscreen_attachment[1])
            });

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

        auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_time;
        last_time = current_time;

        /*directional_lights[0].direction = glm::normalize(glm::vec3(
            0.25f * glm::cos(current_time * 0.5f),
            1.0f,
            0.25f * glm::sin(current_time * 0.5f)));*/

        auto indirect_groups = group_indirect_commands(meshes);
        {
            auto indirect_offset = 0_u32;
            for (auto& group : indirect_groups) {
                auto indirect_commands = std::vector<draw_elements_indirect_t>();
                indirect_commands.reserve(group.second.size());
                for (auto& mesh : group.second) {
                    auto& u_mesh = mesh.get();
                    indirect_commands.emplace_back(draw_elements_indirect_t {
                        static_cast<iris::uint32>(u_mesh.index_count),
                        1,
                        static_cast<iris::uint32>(u_mesh.index_offset),
                        static_cast<iris::int32>(u_mesh.vertex_offset),
                        0
                    });
                }
                ibo.write(indirect_commands.data(), iris::size_bytes(indirect_commands), indirect_offset);
                indirect_offset += iris::size_bytes(indirect_commands);
            }
        }

        auto object_infos = std::vector<object_info_t>();
        object_infos.reserve(meshes.size());
        {
            auto model_index = 0_u32;
            auto mesh_index = 0_u32;
            auto texture_offset = 0_u32;
            for (const auto& model : models) {
                for (const auto& mesh : model.meshes()) {
                    object_infos.emplace_back(object_info_t {
                        mesh_index,
                        model_index,
                        mesh.diffuse_texture + texture_offset,
                        mesh.normal_texture + texture_offset,
                        mesh.specular_texture + texture_offset,
                    });
                    mesh_index++;
                }
                model_index++;
                texture_offset += model.textures().size();
            }
        }

        auto texture_handles = std::vector<iris::uint64>();
        texture_handles.reserve(meshes.size());
        for (const auto& model : models) {
            std::ranges::for_each(model.textures(), [&texture_handles](const auto& texture) {
                texture_handles.emplace_back(texture.handle());
            });
        }

        auto global_pv = calculate_global_projection(camera, directional_lights[0].direction);

        camera_buffer.write(iris::as_const_ptr(camera_data_t {
            camera.projection(),
            camera.view(),
            camera.projection() * camera.view(),
            camera.position(),
            camera.near(),
            camera.far()
        }), sizeof(camera_data_t));

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

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LEQUAL);

        glViewport(0, 0, window.width, window.height);
        depth_only_fbo.clear_depth(1.0f);
        depth_only_fbo.bind();
        depth_only_shader.bind();
        camera_buffer.bind_base(0);
        local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
        global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
        object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
        ibo.bind();
        {
            auto indirect_offset = 0_u32;
            auto object_offset = 0_u32;
            for (auto& [_, group] : indirect_groups) {
                auto& mesh = group.back().get();
                depth_only_shader.set(0, { object_offset });
                glBindVertexArray(mesh.vao);
                glVertexArrayVertexBuffer(mesh.vao, 0, mesh.vbo, 0, mesh.vertex_size);
                glVertexArrayElementBuffer(mesh.vao, mesh.ebo);
                glMultiDrawElementsIndirect(
                    GL_TRIANGLES,
                    GL_UNSIGNED_INT,
                    reinterpret_cast<const void*>(indirect_offset),
                    static_cast<iris::int32>(group.size()),
                    0);
                indirect_offset += group.size() * sizeof(draw_elements_indirect_t);
                object_offset += group.size();
            }
        }

        depth_reduce_init_shader.bind();
        offscreen_attachment[1].bind_texture(0);
        depth_reduce_attachments[0].bind_image_texture(0, 0, false, 0, GL_WRITE_ONLY);
        camera_buffer.bind_base(1);
        glDispatchCompute(depth_reduce_wgc[0].x, depth_reduce_wgc[0].y, 1);

        depth_reduce_shader.bind();
        for (auto i = 1; i < depth_reduce_wgc.size(); i++) {
            depth_reduce_attachments[i - 1].bind_image_texture(0, 0, false, 0, GL_READ_ONLY);
            depth_reduce_attachments[i].bind_image_texture(1, 0, false, 0, GL_WRITE_ONLY);
            glDispatchCompute(depth_reduce_wgc[i].x, depth_reduce_wgc[i].y, 1);
        }

        setup_cascades_shader.bind();
        depth_reduce_attachments.back().bind_image_texture(0, 0, false, 0, GL_READ_ONLY);
        cascade_setup_buffer.bind_base(1);
        camera_buffer.bind_base(2);
        cascade_buffer.bind_base(3);
        glDispatchCompute(1, 1, 1);


        //glCullFace(GL_FRONT);
        glViewport(0, 0, shadow_attachment.width(), shadow_attachment.height());
        shadow_fbo.bind();
        shadow_shader.bind();
        cascade_buffer.bind_base(0);
        local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
        global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
        object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
        texture_buffer.bind_range(4, 0, iris::size_bytes(texture_handles));
        ibo.bind();
        for (auto layer = 0_u32; layer < CASCADE_COUNT; layer++) {
            shadow_shader.set(0, { layer });
            shadow_fbo.set_layer(0, layer);
            shadow_fbo.clear_depth(1.0f);
            auto indirect_offset = 0_u32;
            auto object_offset = 0_u32;
            for (auto& [_, group] : indirect_groups) {
                auto& mesh = group.back().get();
                shadow_shader.set(1, { object_offset });
                glBindVertexArray(mesh.vao);
                glVertexArrayVertexBuffer(mesh.vao, 0, mesh.vbo, 0, mesh.vertex_size);
                glVertexArrayElementBuffer(mesh.vao, mesh.ebo);
                glMultiDrawElementsIndirect(
                    GL_TRIANGLES,
                    GL_UNSIGNED_INT,
                    reinterpret_cast<const void*>(indirect_offset),
                    static_cast<iris::int32>(group.size()),
                    0);
                indirect_offset += group.size() * sizeof(draw_elements_indirect_t);
                object_offset += group.size();
            }
        }
        //glCullFace(GL_BACK);

        glViewport(0, 0, window.width, window.height);
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_EQUAL);

        offscreen_fbo.clear_color(0, { 0_u32, 0_u32, 0_u32, 255_u32 });
        offscreen_fbo.bind();
        main_shader.bind();
        camera_buffer.bind_base(0);
        local_transform_buffer.bind_range(1, 0, iris::size_bytes(local_transforms));
        global_transform_buffer.bind_range(2, 0, iris::size_bytes(global_transforms));
        object_info_buffer.bind_range(3, 0, iris::size_bytes(object_infos));
        directional_lights_buffer.bind_range(4, 0, iris::size_bytes(directional_lights));
        texture_buffer.bind_range(5, 0, iris::size_bytes(texture_handles));
        cascade_buffer.bind_base(6);
        shadow_attachment.bind_texture(0);
        blue_noise_texture.bind(1);
        ibo.bind();
        {
            auto indirect_offset = 0_u32;
            auto object_offset = 0_u32;
            main_shader
                .set(1, { 0_i32 })
                .set(2, { 1_i32 });
            for (auto& [_, group] : indirect_groups) {
                auto& mesh = group.back().get();
                main_shader.set(0, { object_offset });
                glBindVertexArray(mesh.vao);
                glVertexArrayVertexBuffer(mesh.vao, 0, mesh.vbo, 0, mesh.vertex_size);
                glVertexArrayElementBuffer(mesh.vao, mesh.ebo);
                glMultiDrawElementsIndirect(
                    GL_TRIANGLES,
                    GL_UNSIGNED_INT,
                    reinterpret_cast<const void*>(indirect_offset),
                    static_cast<iris::int32>(group.size()),
                    0);
                indirect_offset += group.size() * sizeof(draw_elements_indirect_t);
                object_offset += group.size();
            }
        }

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        fullscreen_shader.bind();
        glBindTextureUnit(0, offscreen_attachment[0].id());
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window.handle);
        glfwPollEvents();

        window.update();
        camera.update(delta_time);
    }
    return 0;
}