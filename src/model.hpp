#pragma once

#include <utilities.hpp>
#include <mesh_pool.hpp>

#include <glm/mat4x4.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>
#include <span>

namespace iris {
    struct mesh_t;
    class mesh_pool_t;

    struct aabb_t {
        alignas(16) glm::vec3 min = {};
        alignas(16) glm::vec3 max = {};
        alignas(16) glm::vec3 center = {};
        alignas(16) glm::vec3 extent = {};
    };

    struct object_t {
        mesh_t mesh = {};
        aabb_t aabb = {};
        glm::vec4 sphere = {};

        uint32 diffuse_texture = 0;
        uint32 normal_texture = 0;
        uint32 specular_texture = 0;
    };

    class model_t {
    public:
        using self = model_t;

        model_t() noexcept;
        ~model_t() noexcept;

        model_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        model_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(mesh_pool_t& mesh_pool, const fs::path& path) noexcept -> self;

        auto objects() const noexcept -> std::span<const object_t>;
        auto transforms() const noexcept -> std::span<const glm::mat4>;
        auto textures() const noexcept -> std::span<const texture_t>;

        auto swap(self& other) noexcept -> void;

    private:
        std::vector<object_t> _objects;
        std::vector<glm::mat4> _transforms;
        std::vector<texture_t> _textures;
    };
} // namespace iris
