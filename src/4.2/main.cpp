#include <utilities.hpp>
#include <framebuffer.hpp>
#include <shader.hpp>
#include <camera.hpp>
#include <buffer.hpp>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <optional>
#include <random>
#include <future>
#include <vector>
#include <thread>
#include <array>
#include <span>

using namespace iris::literals;

struct camera_data_t {
    glm::mat4 inv_pv;
};

enum hittable_type_t : iris::uint32 {
    hittable_type_none = 0,
    hittable_type_sphere = 1,
    hittable_type_triangle = 2,
};

struct hittable_t {
    hittable_type_t type = {};
};

struct sphere_t {
    hittable_t hittable = {};
    glm::vec3 center = {};
    iris::float32 radius = 0;
    iris::uint32 material_id = 0;
};

struct _proxy_hittable_t {
    // should be the max size of all hittable types
    iris::uint32 _data[8];
};

enum material_type_t : iris::uint32 {
    material_type_none = 0,
    material_type_lambertian = 1,
    material_type_metal = 2,
    material_type_dielectric = 3,
};

struct material_t {
    material_type_t type = {};
};

struct lambertian_t {
    material_t material = {};
    glm::vec3 albedo = {};
    glm::vec3 emissive = {};
    iris::float32 e_strength = 0.0f;
};

struct metal_t {
    material_t material = {};
    glm::vec3 albedo = {};
    iris::float32 fuzz = 0;
};

struct dielectric_t {
    material_t material = {};
    iris::float32 refr_index = 0;
};

struct _proxy_material_t {
    // should be the max size of all material types
    iris::uint32 _data[8];
};

int main() {
    if (!glfwInit()) {
        std::cout << "failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window = iris::window_t();
    window.handle = glfwCreateWindow(1920, 1080, "Raytracer", nullptr, nullptr);
    window.width = 1920;
    window.height = 1080;
    if (!window.handle) {
        std::cout << "failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwSetWindowUserPointer(window.handle, &window);

    auto terminate_glfw = iris::defer_t([window]() {
        glfwDestroyWindow(window.handle);
        glfwTerminate();
    });

    glfwMakeContextCurrent(window.handle);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cout << "failed to initialize GLAD" << std::endl;
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

    auto average_shader = iris::shader_t::create("../shaders/4.2/average.vert", "../shaders/4.2/average.frag");
    auto trace_shader = iris::shader_t::create("../shaders/4.2/trace.vert", "../shaders/4.2/trace.frag");

    // empty VAO
    auto vao = 0_u32;
    glGenVertexArrays(1, &vao);

    glViewport(0, 0, window.width, window.height);
    glfwSetFramebufferSizeCallback(window.handle, [](GLFWwindow* handle, int width, int height) {
        auto& window = *static_cast<iris::window_t*>(glfwGetWindowUserPointer(handle));
        window.width = width;
        window.height = height;
        window.is_resized = true;
        glViewport(0, 0, width, height);
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

    auto color_attachment = iris::framebuffer_attachment_t::create(window.width, window.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    auto color_framebuffer = iris::framebuffer_t::create({
        std::cref(color_attachment)
    });

    auto old_color = iris::framebuffer_attachment_t::create(window.width, window.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    auto old_color_framebuffer = iris::framebuffer_t::create({
        std::cref(old_color)
    });

    // uniform buffer
    auto camera_data = camera_data_t();
    auto camera_buffer = iris::buffer_t::create(iris::size_bytes(camera_data), GL_UNIFORM_BUFFER);

    // object buffer
    auto sphere = sphere_t();

    auto hittables = std::vector<_proxy_hittable_t>();
    hittables.resize(16384);

    sphere.hittable.type = hittable_type_sphere;
    sphere.center = glm::vec3(0.0f, 0.0f, -1.0f);
    sphere.radius = 0.5f;
    sphere.material_id = 0;
    std::memcpy(&hittables[0]._data[0], &sphere, sizeof(sphere_t));

    sphere.hittable.type = hittable_type_sphere;
    sphere.center = glm::vec3(0.0f, -100.5f, -1.0f);
    sphere.radius = 100.0f;
    sphere.material_id = 1;
    std::memcpy(&hittables[1]._data[0], &sphere, sizeof(sphere_t));

    sphere.hittable.type = hittable_type_sphere;
    sphere.center = glm::vec3(-1.025, 0.0, -1.0125);
    sphere.radius = 0.5f;
    sphere.material_id = 2;
    std::memcpy(&hittables[2]._data[0], &sphere, sizeof(sphere_t));

    sphere.hittable.type = hittable_type_sphere;
    sphere.center = glm::vec3(1.0125, 0.0, -1.035);
    sphere.radius = 0.5f;
    sphere.material_id = 3;
    std::memcpy(&hittables[3]._data[0], &sphere, sizeof(sphere_t));

    sphere.hittable.type = hittable_type_sphere;
    sphere.center = glm::vec3(2.0625, 0.0, -1.035);
    sphere.radius = 0.5f;
    sphere.material_id = 4;
    std::memcpy(&hittables[4]._data[0], &sphere, sizeof(sphere_t));

    sphere.hittable.type = hittable_type_sphere;
    sphere.center = glm::vec3(520.0, 35.0, 230.0);
    sphere.radius = 400.0f;
    sphere.material_id = 5;
    std::memcpy(&hittables[5]._data[0], &sphere, sizeof(sphere_t));

    auto object_buffer = iris::buffer_t::create(iris::size_bytes(hittables), GL_SHADER_STORAGE_BUFFER);
    object_buffer.write(hittables.data(), iris::size_bytes(hittables));

    auto materials = std::vector<_proxy_material_t>();
    materials.resize(16384);

    auto lambertian = lambertian_t();
    auto metal = metal_t();
    auto dielectric = dielectric_t();

    lambertian.material.type = material_type_lambertian;
    lambertian.albedo = glm::vec3(0.1, 0.2, 0.7);
    std::memcpy(&materials[0]._data[0], &lambertian, iris::size_bytes(lambertian));

    lambertian.material.type = material_type_lambertian;
    lambertian.albedo = glm::vec3(0.8, 0.8, 0.0);
    std::memcpy(&materials[1]._data[0], &lambertian, iris::size_bytes(lambertian));

    lambertian.material.type = material_type_lambertian;
    lambertian.albedo = glm::vec3(1.0);
    std::memcpy(&materials[2]._data[0], &lambertian, iris::size_bytes(lambertian));

    lambertian.material.type = material_type_lambertian;
    lambertian.albedo = glm::vec3(0.6, 0.2, 0.1);
    std::memcpy(&materials[3]._data[0], &lambertian, iris::size_bytes(lambertian));

    lambertian.material.type = material_type_lambertian;
    lambertian.albedo = glm::vec3(0.2, 0.8, 0.2);
    std::memcpy(&materials[4]._data[0], &lambertian, iris::size_bytes(lambertian));

    lambertian.material.type = material_type_lambertian;
    lambertian.albedo = glm::vec3(1.0f);
    lambertian.emissive = glm::vec3(1.0f);
    lambertian.e_strength = 4.0f;
    std::memcpy(&materials[5]._data[0], &lambertian, iris::size_bytes(lambertian));

    auto material_buffer = iris::buffer_t::create(iris::size_bytes(materials), GL_SHADER_STORAGE_BUFFER);
    material_buffer.write(materials.data(), iris::size_bytes(materials));

    auto fps_camera = iris::camera_t(window);
    auto old_camera_position = fps_camera.position();

    auto frame = 0_u32;
    auto last_time = 0.0f;
    auto delta_time = 0.0f;

    while (!glfwWindowShouldClose(window.handle)) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        const auto current_time = static_cast<iris::float32>(glfwGetTime());
        delta_time = current_time - last_time;
        last_time = current_time;

        if (window.is_resized) {
            window.is_resized = false;
            color_attachment = iris::framebuffer_attachment_t::create(window.width, window.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
            color_framebuffer = iris::framebuffer_t::create({
                std::cref(color_attachment)
            });

            old_color = iris::framebuffer_attachment_t::create(window.width, window.height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
            old_color_framebuffer = iris::framebuffer_t::create({
                std::cref(old_color)
            });

            frame = 0;
        }

        if (window.is_mouse_captured || glm::any(glm::greaterThan(glm::abs(fps_camera.position() - old_camera_position), glm::vec3(FLT_EPSILON)))) {
            frame = 0;
        }
        camera_data.inv_pv = glm::inverse(fps_camera.projection() * fps_camera.view());
        camera_buffer.write(&camera_data, iris::size_bytes(camera_data));

        color_framebuffer.bind();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        trace_shader
            .bind()
            .set(0, glm::vec2(window.width, window.height))
            .set(1, { frame })
            .set(2, { current_time * 1000 + frame });
        camera_buffer.bind_base(0);
        object_buffer.bind_base(1);
        material_buffer.bind_base(2);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, old_color.id());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, color_attachment.id());
        average_shader
            .bind()
            .set(0, { 0_i32 })
            .set(1, { 1_i32 })
            .set(2, { frame });

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, old_color_framebuffer.id());
        glBlitFramebuffer(0, 0, window.width, window.height, 0, 0, window.width, window.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window.handle);
        glfwPollEvents();
        old_camera_position = fps_camera.position();
        window.update();
        fps_camera.update(delta_time);
        ++frame;
    }
    return 0;
}
