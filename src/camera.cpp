#include <camera.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>

namespace iris {
    static auto plane_from_points(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) noexcept -> plane_t {
        auto plane = plane_t();
        plane.normal = glm::normalize(glm::cross(c - a, b - a));
        plane.distance = glm::dot(plane.normal, a);
        return plane;
    }

    void window_t::update() noexcept {
        if (is_mouse_captured) {
            auto c_x = 0.0;
            auto c_y = 0.0;
            glfwGetCursorPos(handle, &c_x, &c_y);
            cursor_position.last_x = cursor_position.x;
            cursor_position.last_y = cursor_position.y;
            if (cursor_position.last_x == 0 && cursor_position.last_y == 0) {
                cursor_position.last_x = static_cast<iris::float32>(c_x);
                cursor_position.last_y = static_cast<iris::float32>(c_y);
            }
            cursor_position.x = static_cast<iris::float32>(c_x);
            cursor_position.y = static_cast<iris::float32>(c_y);
        }
    }

    camera_t::camera_t(const window_t& window) noexcept : _window(std::cref(window)) {}

    auto camera_t::create(const window_t& window) noexcept -> self {
        auto camera = self(window);
        camera.update(0);
        return camera;
    }

    auto camera_t::position() const noexcept -> const glm::vec3& {
        return _position;
    }

    auto camera_t::front() const noexcept -> const glm::vec3& {
        return _front;
    }

    auto camera_t::up() const noexcept -> const glm::vec3& {
        return _up;
    }

    auto camera_t::right() const noexcept -> const glm::vec3& {
        return _right;
    }

    auto camera_t::yaw() const noexcept -> iris::float32 {
        return _yaw;
    }

    auto camera_t::pitch() const noexcept -> iris::float32 {
        return _pitch;
    }

    auto camera_t::fov() const noexcept -> iris::float32 {
        return glm::radians(_fov);
    }

    auto camera_t::aspect() const noexcept -> iris::float32 {
        const auto& window = _window.get();
        const auto aspect =
            static_cast<iris::float32>(window.width) /
            static_cast<iris::float32>(window.height);
        return aspect;
    }

    auto camera_t::near() const noexcept -> iris::float32 {
        return _near;
    }

    auto camera_t::far() const noexcept -> iris::float32 {
        return _far;
    }

    auto camera_t::view() const noexcept -> glm::mat4 {
        return glm::lookAt(_position, _position + _front, _up);
    }

    auto camera_t::projection(bool infinite, bool reverse_z) const noexcept -> glm::mat4 {
        const auto type = ((uint32(infinite) << 1u) | uint32(reverse_z));
        switch (type) {
            case 0b00u:
                return glm::perspective(fov(), aspect(), _near, _far);
            case 0b01u:
                return glm::perspective(fov(), aspect(), _far, _near);
            case 0b10u:
                return glm::infinitePerspective(fov(), aspect(), _near);
            case 0b11u:
                const auto f = 1.0f / glm::tan(fov() / 2.0f);
                return {
                    f / aspect(), 0.0f, 0.0f, 0.0f,
                    0.0f, f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, -1.0f,
                    0.0f, 0.0f, _near, 0.0f
                };
        }
        return {};
    }

    auto camera_t::update(iris::float32 dt) noexcept -> void {
        constexpr auto sensitivity = 0.1f;
        const auto& window = _window.get();
        const auto speed = 7.5f * dt;

        // mouse input handling (rely on window_t to zero cursor position when mouse is not captured)
        const auto& [last_c_x, last_c_y, c_x, c_y] = window.cursor_position;
        const auto dx = c_x - last_c_x;
        const auto dy = last_c_y - c_y;
        _yaw += sensitivity * dx;
        _pitch += sensitivity * dy;
        if (_pitch > 89.9f) {
            _pitch = 89.9f;
        }
        if (_pitch < -89.9f) {
            _pitch = -89.9f;
        }
        const auto r_yaw = glm::radians(_yaw);
        const auto r_pitch = glm::radians(_pitch);

        // keyboard handling
        const auto p_y = _position.y;
        if (glfwGetKey(window.handle, GLFW_KEY_W) == GLFW_PRESS) {
            _position.x += glm::cos(r_yaw) * speed;
            _position.z += glm::sin(r_yaw) * speed;
        }
        if (glfwGetKey(window.handle, GLFW_KEY_S) == GLFW_PRESS) {
            _position.x -= glm::cos(r_yaw) * speed;
            _position.z -= glm::sin(r_yaw) * speed;
        }
        if (glfwGetKey(window.handle, GLFW_KEY_D) == GLFW_PRESS) {
            _position += speed * _right;
        }
        if (glfwGetKey(window.handle, GLFW_KEY_A) == GLFW_PRESS) {
            _position -= speed * _right;
        }
        _position.y = p_y;
        if (glfwGetKey(window.handle, GLFW_KEY_SPACE) == GLFW_PRESS) {
            _position.y += speed;
        }
        if (glfwGetKey(window.handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            _position.y -= speed;
        }

        _front = glm::normalize(glm::vec3(
            // x-axis is dependent on: angle (cos(r_pitch)) of y-axis against xz plane and angle (r_yaw) of z-axis against x.
            glm::cos(r_yaw) * glm::cos(r_pitch),
            // y-axis is dependent on: angle (r_pitch) of y-axis against xz plane.
            glm::sin(r_pitch),
            // z-axis is dependent on: angle (sin(r_pitch)) of y-axis against xz plane and angle (r_yaw) of z-axis against x.
            glm::sin(r_yaw) * glm::cos(r_pitch)));
        // gram-schmidt orthogonalization: use the world up vector, orthogonal to the front vector, to get the right vector.
        _right = glm::normalize(glm::cross(_front, { 0.0f, 1.0f, 0.0f }));
        // use the right vector, orthogonal to the front vector, to get the actual up vector.
        _up = glm::normalize(glm::cross(_right, _front));
    }

    auto make_perspective_frustum(const glm::mat4& pv) noexcept -> frustum_t {
        auto frustum = frustum_t();
        auto planes = std::array<glm::vec4, 6>();
        for (auto i = 0_u32; i < 4; ++i) { planes[0][i] = pv[i][3] + pv[i][0]; }
        for (auto i = 0_u32; i < 4; ++i) { planes[1][i] = pv[i][3] - pv[i][0]; }
        for (auto i = 0_u32; i < 4; ++i) { planes[2][i] = pv[i][3] + pv[i][1]; }
        for (auto i = 0_u32; i < 4; ++i) { planes[3][i] = pv[i][3] - pv[i][1]; }
        for (auto i = 0_u32; i < 4; ++i) { planes[4][i] = pv[i][3] + pv[i][2]; }
        for (auto i = 0_u32; i < 4; ++i) { planes[5][i] = pv[i][3] - pv[i][2]; }

        for (auto i = 0_u32; i < 6; ++i) {
            const auto& plane = glm::normalize(planes[i]);
            frustum.planes[i].normal = glm::vec3(plane);
            frustum.planes[i].distance = -plane.w;
        }
        return frustum;
    }
} // namespace iris
