#pragma once

#include <utilities.hpp>
#include <allocator.hpp>

#include <glad/gl.h>

#include <vector>

namespace iris {
    struct mesh_t {
        uint64 vertex_offset = 0;
        uint64 index_offset = 0;
        uint64 index_count = 0;
        uint64 vertex_size = 0;

        buffer_slice_t vertex_slice;
        buffer_slice_t index_slice;

        uint32 vao = 0;
        uint32 vbo = 0;
        uint32 ebo = 0;
    };

    // assumed all components are type = GL_FLOAT and binding = 0
    struct vertex_attribute_t {
        uint32 index = 0;
        uint32 components = 0;
    };

    class mesh_pool_t {
    public:
        using self = mesh_pool_t;

        mesh_pool_t() noexcept;
        ~mesh_pool_t() noexcept;

        mesh_pool_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        mesh_pool_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create() noexcept -> self;

        template <typename T>
        auto make_mesh(
            const std::vector<T>& vertices,
            const std::vector<uint32>& indices,
            const std::vector<vertex_attribute_t>& vertex_format) noexcept -> mesh_t;

        auto swap(self& other) noexcept -> void;

    private:
        struct _vertex_buffer_package {
            uint32 vao = 0;
            std::vector<uint32> vbos;
            allocator_t allocator;
        };

        std::unordered_map<uint64, _vertex_buffer_package> _vbps;
        std::vector<uint32> _ebos;
        allocator_t allocator;
    };

    template <typename T>
    auto mesh_pool_t::make_mesh(const std::vector<T>& vertices,
                                const std::vector<uint32>& indices,
                                const std::vector<vertex_attribute_t>& vertex_format) noexcept -> mesh_t {
        const auto vertex_size = sizeof(T);
        const auto index_count = indices.size();

        // either insert a new VAO + VBO or fetch from cache
        auto& vbp = _vbps[vertex_size];
        if (!vbp.vao) {
            vbp.allocator = allocator_t::create(2_GiB);

            glCreateVertexArrays(1, &vbp.vao);
            auto offset = 0_u32;
            for (auto& attribute : vertex_format) {
                glEnableVertexArrayAttrib(vbp.vao, attribute.index);
                glVertexArrayAttribFormat(vbp.vao, attribute.index, attribute.components, GL_FLOAT, GL_FALSE, offset);
                glVertexArrayAttribBinding(vbp.vao, attribute.index, 0);
                offset += attribute.components * sizeof(float32);
            }

            auto& vbo = vbp.vbos.emplace_back();
            glCreateBuffers(1, &vbo);
            glNamedBufferStorage(vbo, vbp.allocator.capacity(), nullptr, GL_DYNAMIC_STORAGE_BIT);
            glVertexArrayVertexBuffer(vbp.vao, 0, vbo, 0, vertex_size);
        }

        // copy vertices
        auto vertex_slice = vbp.allocator.allocate(size_bytes(vertices));
        if (vertex_slice.index() >= vbp.vbos.size()) {
            vbp.vbos.resize(vertex_slice.index() + 1);
        }
        if (!vbp.vbos[vertex_slice.index()]) {
            glCreateBuffers(1, &vbp.vbos[vertex_slice.index()]);
            glNamedBufferStorage(vbp.vbos[vertex_slice.index()], vbp.allocator.capacity(), nullptr, GL_DYNAMIC_STORAGE_BIT);
            glVertexArrayVertexBuffer(vbp.vao, vertex_slice.index(), vbp.vbos[vertex_slice.index()], 0, vertex_size);
        }
        glNamedBufferSubData(vbp.vbos[vertex_slice.index()], vertex_slice.offset(), vertex_slice.size(), vertices.data());

        // copy indices
        auto index_slice = allocator.allocate(size_bytes(indices));
        if (index_slice.index() >= _ebos.size()) {
            _ebos.resize(index_slice.index() + 1);
        }
        if (!_ebos[index_slice.index()]) {
            glCreateBuffers(1, &_ebos[index_slice.index()]);
            glNamedBufferStorage(_ebos[index_slice.index()], allocator.capacity(), nullptr, GL_DYNAMIC_STORAGE_BIT);
            glVertexArrayElementBuffer(vbp.vao, _ebos[index_slice.index()]);
        }
        glNamedBufferSubData(_ebos[index_slice.index()], index_slice.offset(), index_slice.size(), indices.data());

        auto mesh = mesh_t();
        mesh.vertex_offset = vertex_slice.offset() / vertex_size;
        mesh.index_offset = index_slice.offset() / sizeof(uint32);
        mesh.index_count = index_count;
        mesh.vertex_size = vertex_size;

        mesh.vao = vbp.vao;
        mesh.vbo = vbp.vbos[vertex_slice.index()];
        mesh.ebo = _ebos[index_slice.index()];

        mesh.vertex_slice = std::move(vertex_slice);
        mesh.index_slice = std::move(index_slice);
        return mesh;
    }
} // namespace iris
