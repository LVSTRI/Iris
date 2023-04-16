#pragma once

#include <utilities.hpp>

#include <glad/gl.h>

#include <glm/mat4x4.hpp>

#include <cassert>
#include <span>

namespace iris {
    class shader_t {
    public:
        using self = shader_t;

        shader_t() noexcept;
        ~shader_t() noexcept;

        shader_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        shader_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(const fs::path& vertex, const fs::path& fragment) noexcept -> self;
        static auto create_compute(const fs::path& compute) noexcept -> self;

        auto bind() const noexcept -> const self&;

        auto id() const noexcept -> uint32;

        template <uint64 N>
        auto set(int32 location, const int32(&values)[N]) const noexcept -> const self&;
        template <uint64 N>
        auto set(int32 location, const uint32(&values)[N]) const noexcept -> const self&;
        template <uint64 N>
        auto set(int32 location, const float32(&values)[N]) const noexcept -> const self&;

        auto set(int32 location, const glm::vec2& value) const noexcept -> const self&;
        auto set(int32 location, const glm::vec3& value) const noexcept -> const self&;
        auto set(int32 location, const glm::vec4& value) const noexcept -> const self&;

        template <uint64 N>
        auto set(int32 location, std::span<const int32(&)[N]> values) const noexcept -> const self&;
        template <uint64 N>
        auto set(int32 location, std::span<const uint32(&)[N]> values) const noexcept -> const self&;
        template <uint64 N>
        auto set(int32 location, std::span<const float32(&)[N]> values) const noexcept -> const self&;

        auto set(int32 location, const glm::mat4& values) const noexcept -> const self&;
        auto set(int32 location, std::span<const glm::mat4> values) const noexcept -> const self&;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _id = 0;
    };

    template <uint64 N>
    auto shader_t::set(int32 location, const int32(&values)[N]) const noexcept -> const self& {
        switch (N) {
            case 1:
                glUniform1i(location, values[0]);
                break;
            case 2:
                glUniform2i(location, values[0], values[1]);
                break;
            case 3:
                glUniform3i(location, values[0], values[1], values[2]);
                break;
            case 4:
                glUniform4i(location, values[0], values[1], values[2], values[3]);
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }

    template <uint64 N>
    auto shader_t::set(int32 location, const uint32(&values)[N]) const noexcept -> const self& {
        switch (N) {
            case 1:
                glUniform1ui(location, values[0]);
                break;
            case 2:
                glUniform2ui(location, values[0], values[1]);
                break;
            case 3:
                glUniform3ui(location, values[0], values[1], values[2]);
                break;
            case 4:
                glUniform4ui(location, values[0], values[1], values[2], values[3]);
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }

    template <uint64 N>
    auto shader_t::set(int32 location, const float32(&values)[N]) const noexcept -> const self& {
        switch (N) {
            case 1:
                glUniform1f(location, values[0]);
                break;
            case 2:
                glUniform2f(location, values[0], values[1]);
                break;
            case 3:
                glUniform3f(location, values[0], values[1], values[2]);
                break;
            case 4:
                glUniform4f(location, values[0], values[1], values[2], values[3]);
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }

    template <uint64 N>
    auto shader_t::set(int32 location, std::span<const int32(&)[N]> values) const noexcept -> const self& {
        switch (N) {
            case 1:
                glUniform1iv(location, values.size(), values.data());
                break;
            case 2:
                glUniform2iv(location, values.size(), values.data());
                break;
            case 3:
                glUniform3iv(location, values.size(), values.data());
                break;
            case 4:
                glUniform4iv(location, values.size(), values.data());
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }

    template <uint64 N>
    auto shader_t::set(int32 location, std::span<const uint32(&)[N]> values) const noexcept -> const self& {
        switch (N) {
            case 1:
                glUniform1uiv(location, values.size(), values.data());
                break;
            case 2:
                glUniform2uiv(location, values.size(), values.data());
                break;
            case 3:
                glUniform3uiv(location, values.size(), values.data());
                break;
            case 4:
                glUniform4uiv(location, values.size(), values.data());
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }

    template <uint64 N>
    auto shader_t::set(int32 location, std::span<const float32(&)[N]> values) const noexcept -> const self& {
        switch (N) {
            case 1:
                glUniform1fv(location, values.size(), values.data());
                break;
            case 2:
                glUniform2fv(location, values.size(), values.data());
                break;
            case 3:
                glUniform3fv(location, values.size(), values.data());
                break;
            case 4:
                glUniform4fv(location, values.size(), values.data());
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }
} // namespace iris
