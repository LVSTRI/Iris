#include <shader.hpp>

#include <glad/gl.h>

#include <glm/gtc/type_ptr.hpp>

#include <array>

namespace iris {
    static auto shader_compile_status(iris::uint32 shader) -> void {
        auto success = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            auto info = std::array<char, 1024>();
            glGetShaderInfoLog(shader, info.size(), nullptr, info.data());
            iris::log("err: shader compilation failed with: ", info.data());
        }
    }

    static auto program_link_status(iris::uint32 program) -> void {
        auto success = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            auto info = std::array<char, 1024>();
            glGetProgramInfoLog(program, info.size(), nullptr, info.data());
            iris::log("err: shader program linking failed with: ", info.data());
        }
    }

    shader_t::shader_t() noexcept = default;

    shader_t::~shader_t() noexcept {
        glDeleteProgram(_id);
    }

    shader_t::shader_t(self&& other) noexcept {
        swap(other);
    }

    auto shader_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto shader_t::create(const fs::path& vertex, const fs::path& fragment) noexcept -> self {
        auto shader = self();
        shader._id = glCreateProgram();
        auto vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        const auto vertex_shader_file = iris::whole_file(vertex);
        const auto* vertex_shader_source = vertex_shader_file.c_str();
        glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
        glCompileShader(vertex_shader);
        shader_compile_status(vertex_shader);

        auto fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        const auto fragment_shader_file = iris::whole_file(fragment);
        const auto* fragment_shader_source = fragment_shader_file.c_str();
        glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
        glCompileShader(fragment_shader);
        shader_compile_status(fragment_shader);

        glAttachShader(shader._id, vertex_shader);
        glAttachShader(shader._id, fragment_shader);
        glLinkProgram(shader._id);
        program_link_status(shader._id);

        glDeleteShader(fragment_shader);
        glDeleteShader(vertex_shader);
        return shader;
    }

    auto shader_t::bind() const noexcept -> const self& {
        glUseProgram(_id);
        return *this;
    }

    auto shader_t::id() const noexcept -> uint32 {
        return _id;
    }

    auto shader_t::set(int32 location, const glm::vec2& value) const noexcept -> const self& {
        glUniform2f(location, value[0], value[1]);
        return *this;
    }

    auto shader_t::set(int32 location, const glm::vec3& value) const noexcept -> const self& {
        glUniform3f(location, value[0], value[1], value[2]);
        return *this;
    }

    auto shader_t::set(int32 location, const glm::vec4& value) const noexcept -> const self& {
        glUniform4f(location, value[0], value[1], value[2], value[3]);
        return *this;
    }

    auto shader_t::set(int32 location, const glm::mat4& values) const noexcept -> const self& {
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(values));
        return *this;
    }

    auto shader_t::set(int32 location, std::span<const glm::mat4> values) const noexcept -> const self& {
        glUniformMatrix4fv(location, values.size(), GL_FALSE, glm::value_ptr(values[0]));
        return *this;
    }

    auto shader_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
    }
} // namespace iris
