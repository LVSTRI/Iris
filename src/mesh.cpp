#include <texture.hpp>

#include <mesh.hpp>

#include <glad/gl.h>

#include <numeric>

namespace iris {
    mesh_t::mesh_t() noexcept = default;

    mesh_t::~mesh_t() noexcept {
        glDeleteVertexArrays(1, &_vao);
        glDeleteBuffers(1, &_vbo);
        glDeleteBuffers(1, &_ebo);
    }

    mesh_t::mesh_t(self&& other) noexcept {
        swap(other);
    }

    auto mesh_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto mesh_t::create(std::span<const vertex_t> vertices,
                        std::span<const uint32> indices,
                        std::vector<std::reference_wrapper<const texture_t>> textures,
                        glm::mat4 transform) noexcept -> self {
        auto mesh = self();

        glGenVertexArrays(1, &mesh._vao);
        glGenBuffers(1, &mesh._vbo);
        glGenBuffers(1, &mesh._ebo);

        auto indices_count = indices.size();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh._ebo);
        if (indices.empty()) {
            indices_count = vertices.size();
            auto g_indices = std::vector<uint32>(vertices.size());
            std::iota(g_indices.begin(), g_indices.end(), 0_u32);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, size_bytes(g_indices), g_indices.data(), GL_STATIC_DRAW);
        } else {
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size_bytes(), indices.data(), GL_STATIC_DRAW);
        }

        glBindVertexArray(mesh._vao);
        glBindBuffer(GL_ARRAY_BUFFER, mesh._vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size_bytes(), vertices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, position));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, uv));

        mesh._textures = std::move(textures);

        mesh._transform = transform;

        auto aabb = aabb_t();
        aabb.min = glm::vec3(std::numeric_limits<glm::vec3::value_type>::max());
        aabb.max = glm::vec3(std::numeric_limits<glm::vec3::value_type>::lowest());
        for (const auto& vertex : vertices) {
            aabb.min = glm::min(aabb.min, vertex.position);
            aabb.max = glm::max(aabb.max, vertex.position);
        }
        aabb.center = (aabb.min + aabb.max) / 2.0f;
        aabb.size = aabb.max - aabb.min;
        mesh._aabb = aabb;

        mesh._vertices = vertices.size();
        mesh._indices = indices_count;

        return mesh;
    }

    auto mesh_t::vao() const noexcept -> uint32 {
        return _vao;
    }

    auto mesh_t::vbo() const noexcept -> uint32 {
        return _vbo;
    }

    auto mesh_t::ebo() const noexcept -> uint32 {
        return _ebo;
    }

    auto mesh_t::transform() const noexcept -> const glm::mat4& {
        return _transform;
    }

    auto mesh_t::aabb() const noexcept -> const aabb_t& {
        return _aabb;
    }

    auto mesh_t::textures() const noexcept -> std::span<const std::reference_wrapper<const texture_t>> {
        return _textures;
    }

    auto mesh_t::vertices() const noexcept -> uint32 {
        return _vertices;
    }

    auto mesh_t::indices() const noexcept -> uint32 {
        return _indices;
    }

    auto mesh_t::draw() const noexcept -> void {
        glBindVertexArray(_vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
        glDrawElements(GL_TRIANGLES, _indices, GL_UNSIGNED_INT, nullptr);
    }

    auto mesh_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_vao, other._vao);
        swap(_vbo, other._vbo);
        swap(_ebo, other._ebo);
        swap(_transform, other._transform);
        swap(_aabb, other._aabb);
        swap(_textures, other._textures);
        swap(_vertices, other._vertices);
        swap(_indices, other._indices);
    }
} // namespace iris
