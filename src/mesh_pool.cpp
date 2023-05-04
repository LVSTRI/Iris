#include <mesh_pool.hpp>

namespace iris {
    mesh_pool_t::mesh_pool_t() noexcept = default;

    mesh_pool_t::~mesh_pool_t() noexcept {
        for (auto& [_, vbp] : _vbps) {
            glDeleteVertexArrays(1, &vbp.vao);
            glDeleteBuffers(vbp.vbos.size(), vbp.vbos.data());
        }

        glDeleteBuffers(_ebos.size(), _ebos.data());
    }

    mesh_pool_t::mesh_pool_t(self&& other) noexcept {
        swap(other);
    }

    auto mesh_pool_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto mesh_pool_t::create() noexcept -> self {
        auto mesh_pool = self();
        mesh_pool.allocator = allocator_t::create();
        return mesh_pool;
    }

    auto mesh_pool_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_vbps, other._vbps);
        swap(_ebos, other._ebos);
        swap(allocator, other.allocator);
    }
} // namespace iris
