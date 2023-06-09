#pragma once

#include <GLFW/glfw3.h>

#include <utilities.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace iris {
    struct cursor_position_t {
        iris::float32 last_x = 0;
        iris::float32 last_y = 0;
        iris::float32 x = 0;
        iris::float32 y = 0;
    };

    struct window_t {
        GLFWwindow* handle = nullptr;
        iris::int32 width = 0;
        iris::int32 height = 0;

        cursor_position_t cursor_position = {};
        bool is_mouse_captured = false;
        bool is_focused = true;
        bool is_resized = false;

        void update() noexcept;
    };

    struct alignas(16) plane_t {
        glm::vec3 normal = {};
        iris::float32 distance = 0;

        plane_t() noexcept = default;

        plane_t(const glm::vec3& n, const glm::vec3& p) noexcept
            : normal(glm::normalize(n)),
              distance(glm::dot(normal, p)) {}
    };

    struct frustum_t {
        plane_t planes[6];
    };

    class camera_t {
    public:
        using self = camera_t;

        camera_t(const window_t& window) noexcept;
        ~camera_t() noexcept = default;

        camera_t(const self&) noexcept = default;
        auto operator =(const self&) noexcept -> self& = default;
        camera_t(self&&) noexcept = default;
        auto operator =(self&&) noexcept -> self& = default;

        static auto create(const window_t& window) noexcept -> self;

        auto position() const noexcept -> const glm::vec3&;
        auto front() const noexcept -> const glm::vec3&;
        auto up() const noexcept -> const glm::vec3&;
        auto right() const noexcept -> const glm::vec3&;

        auto yaw() const noexcept -> iris::float32;
        auto pitch() const noexcept -> iris::float32;
        auto fov() const noexcept -> iris::float32;
        auto aspect() const noexcept -> iris::float32;
        auto near() const noexcept -> iris::float32;
        auto far() const noexcept -> iris::float32;

        auto view() const noexcept -> glm::mat4;
        auto projection(bool infinite = false, bool reverse_z = false) const noexcept -> glm::mat4;

        auto update(iris::float32 dt) noexcept -> void;

    private:
        glm::vec3 _position = { -7.5f, 1.0f, 0.0f };
        glm::vec3 _front = { 0.0f, 0.0f, -1.0f };
        glm::vec3 _up = { 0.0f, 1.0f, 0.0f };
        glm::vec3 _right = { 1.0f, 0.0f, 0.0f };

        iris::float32 _yaw = 0.0f;
        iris::float32 _pitch = 0.0f;
        iris::float32 _fov = 60.0f;
        iris::float32 _near = 0.1f;
        iris::float32 _far = 512.0f;

        std::reference_wrapper<const window_t> _window;
    };

    auto make_perspective_frustum(const glm::mat4& pv) noexcept -> frustum_t;
} // namespace iris
