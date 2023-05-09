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
        static auto create_mesh(const fs::path& task, const fs::path& mesh, const fs::path& fragment) noexcept -> self;

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
                glProgramUniform1i(_id, location, values[0]);
                break;
            case 2:
                glProgramUniform2i(_id, location, values[0], values[1]);
                break;
            case 3:
                glProgramUniform3i(_id, location, values[0], values[1], values[2]);
                break;
            case 4:
                glProgramUniform4i(_id, location, values[0], values[1], values[2], values[3]);
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
                glProgramUniform1ui(_id, location, values[0]);
                break;
            case 2:
                glProgramUniform2ui(_id, location, values[0], values[1]);
                break;
            case 3:
                glProgramUniform3ui(_id, location, values[0], values[1], values[2]);
                break;
            case 4:
                glProgramUniform4ui(_id, location, values[0], values[1], values[2], values[3]);
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
                glProgramUniform1f(_id, location, values[0]);
                break;
            case 2:
                glProgramUniform2f(_id, location, values[0], values[1]);
                break;
            case 3:
                glProgramUniform3f(_id, location, values[0], values[1], values[2]);
                break;
            case 4:
                glProgramUniform4f(_id, location, values[0], values[1], values[2], values[3]);
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
                glProgramUniform1iv(_id, location, values.size(), values.data());
                break;
            case 2:
                glProgramUniform2iv(_id, location, values.size(), values.data());
                break;
            case 3:
                glProgramUniform3iv(_id, location, values.size(), values.data());
                break;
            case 4:
                glProgramUniform4iv(_id, location, values.size(), values.data());
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
                glProgramUniform1uiv(_id, location, values.size(), values.data());
                break;
            case 2:
                glProgramUniform2uiv(_id, location, values.size(), values.data());
                break;
            case 3:
                glProgramUniform3uiv(_id, location, values.size(), values.data());
                break;
            case 4:
                glProgramUniform4uiv(_id, location, values.size(), values.data());
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
                glProgramUniform1fv(_id, location, values.size(), values.data());
                break;
            case 2:
                glProgramUniform2fv(_id, location, values.size(), values.data());
                break;
            case 3:
                glProgramUniform3fv(_id, location, values.size(), values.data());
                break;
            case 4:
                glProgramUniform4fv(_id, location, values.size(), values.data());
                break;
            default:
                assert(false && "bad size");
        }
        return *this;
    }
} // namespace iris
