#pragma once

#include <utilities.hpp>

#include <glm/glm.hpp>

#include <vector>
#include <span>

namespace iris {
    struct vertex_t {
        glm::vec3 position = {};
        glm::vec3 normal = {};
        glm::vec2 uv = {};
    };

    struct aabb_t {
        glm::vec3 min = {};
        glm::vec3 max = {};
        glm::vec3 center = {};
        glm::vec3 size = {};
    };

    class texture_t;
    class mesh_t {
    public:
        using self = mesh_t;

        mesh_t() noexcept;
        ~mesh_t() noexcept;

        mesh_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        mesh_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(std::vector<vertex_t> vertices,
                           std::vector<uint32> indices,
                           std::vector<std::reference_wrapper<const texture_t>> textures,
                           glm::mat4 transform = glm::mat4(1.0f)) noexcept -> self;

        auto vao() const noexcept -> uint32;
        auto vbo() const noexcept -> uint32;
        auto ebo() const noexcept -> uint32;
        auto transform() const noexcept -> const glm::mat4&;
        auto aabb() const noexcept -> const aabb_t&;
        auto textures() const noexcept -> std::span<const std::reference_wrapper<const texture_t>>;
        auto vertices() const noexcept -> std::span<const vertex_t>;
        auto indices() const noexcept -> std::span<const uint32>;

        auto draw() const noexcept -> void;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _vao = 0;
        uint32 _vbo = 0;
        uint32 _ebo = 0;

        glm::mat4 _transform = glm::mat4(1.0f);
        aabb_t _aabb = {};

        std::vector<std::reference_wrapper<const texture_t>> _textures;
        std::vector<vertex_t> _vertices;
        std::vector<uint32> _indices;
    };
} // namespace iris
