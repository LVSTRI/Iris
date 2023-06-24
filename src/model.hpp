#pragma once

#include <utilities.hpp>
#include <mesh_pool.hpp>

#include <glm/mat4x4.hpp>

#include <meshoptimizer.h>

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

    struct vertex_format_t {
        glm::vec3 position = {};
        glm::vec3 normal = {};
        glm::vec2 uv = {};
        glm::vec4 tangent = {};
    };

    struct meshlet_vertex_format_t {
        alignas(16) glm::vec3 position = {};
        alignas(16) glm::vec3 normal = {};
        alignas(16) glm::vec2 uv = {};
        alignas(16) glm::vec4 tangent = {};
    };

    struct object_t {
        uint32 mesh = 0;
        aabb_t aabb = {};
        glm::vec4 sphere = {};
        glm::vec3 scale = {};

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

        auto acquire_mesh(uint32 index) const noexcept -> const mesh_t&;

        auto swap(self& other) noexcept -> void;

    private:
        std::vector<object_t> _objects;
        std::vector<glm::mat4> _transforms;
        std::vector<texture_t> _textures;
        std::vector<mesh_t> _meshes;
    };

    struct meshlet_t {
        uint32 vertex_offset = 0;
        uint32 index_offset = 0;
        uint32 index_count = 0;
        uint32 triangle_offset = 0;
        uint32 triangle_count = 0;
    };

    struct meshlet_group_t {
        std::vector<meshlet_t> meshlets;
        uint32 vertex_count = 0;
        uint32 vertex_offset = 0;

        uint32 diffuse_index = 0;
        uint32 normal_index = 0;
        uint32 specular_index = 0;
    };

    class meshlet_model_t {
    public:
        using self = meshlet_model_t;

        meshlet_model_t() noexcept;
        ~meshlet_model_t() noexcept;

        meshlet_model_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        meshlet_model_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(const fs::path& path) noexcept -> self;

        auto meshlet_groups() const noexcept -> std::span<const meshlet_group_t>;
        auto transforms() const noexcept -> std::span<const glm::mat4>;
        auto textures() const noexcept -> std::span<const texture_t>;
        auto vertices() const noexcept -> std::span<const meshlet_vertex_format_t>;
        auto indices() const noexcept -> std::span<const uint32>;
        auto triangles() const noexcept -> std::span<const uint8>;
        auto meshlet_count() const noexcept -> uint32;

        auto swap(self& other) noexcept -> void;

    private:
        std::vector<meshlet_group_t> _meshlet_groups;
        std::vector<meshlet_vertex_format_t> _vertices;
        std::vector<uint32> _indices;
        std::vector<uint8> _triangles;

        std::vector<glm::mat4> _transforms;
        std::vector<texture_t> _textures;
        uint32 _meshlet_count = 0;
    };
} // namespace iris
