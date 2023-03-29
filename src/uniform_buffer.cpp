#include <uniform_buffer.hpp>

#include <glad/gl.h>

#include <cassert>

namespace iris {
    uniform_buffer_t::uniform_buffer_t() noexcept = default;

    uniform_buffer_t::~uniform_buffer_t() noexcept {
        glDeleteBuffers(1, &_id);
    }

    uniform_buffer_t::uniform_buffer_t(self&& other) noexcept {
        swap(other);
    }

    auto uniform_buffer_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto uniform_buffer_t::create(uint32 size, bool mapped) noexcept -> self {
        auto buffer = self();
        glGenBuffers(1, &buffer._id);
        glBindBuffer(GL_UNIFORM_BUFFER, buffer._id);
        glBufferData(GL_UNIFORM_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);

        if (mapped) {
            buffer._mapped = glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
        }

        buffer._size = size;
        return buffer;
    }

    auto uniform_buffer_t::id() const noexcept -> uint32 {
        return _id;
    }

    auto uniform_buffer_t::size() const noexcept -> uint64 {
        return _size;
    }

    auto uniform_buffer_t::mapped() const noexcept -> void* {
        return _mapped;
    }

    auto uniform_buffer_t::write(const void* data, uint64 size, uint64 offset) const noexcept -> const self& {
        assert(offset + size <= _size && "overflow");
        glBindBuffer(GL_UNIFORM_BUFFER, _id);
        glBufferSubData(GL_UNIFORM_BUFFER, offset, size, data);
        return *this;
    }

    auto uniform_buffer_t::bind_base(uint32 index) const noexcept -> const self& {
        glBindBuffer(GL_UNIFORM_BUFFER, _id);
        glBindBufferBase(GL_UNIFORM_BUFFER, index, _id);
        return *this;
    }

    auto uniform_buffer_t::bind_range(uint32 index, uint64 offset, uint64 size) const noexcept -> const self& {
        glBindBuffer(GL_UNIFORM_BUFFER, _id);
        glBindBufferRange(GL_UNIFORM_BUFFER, index, _id, offset, size);
        return *this;
    }

    auto uniform_buffer_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
        swap(_size, other._size);
        swap(_mapped, other._mapped);
    }
} // namespace iris
