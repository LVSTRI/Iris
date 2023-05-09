#include <buffer.hpp>

#include <glad/gl.h>

#include <cassert>

namespace iris {
    buffer_t::buffer_t() noexcept = default;

    buffer_t::~buffer_t() noexcept {
        if (_mapped) {
            glUnmapNamedBuffer(_id);
        }
        if (_id) {
            glDeleteBuffers(1, &_id);
        }
    }

    buffer_t::buffer_t(self&& other) noexcept {
        swap(other);
    }

    auto buffer_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto buffer_t::create(uint32 size, uint32 type, uint32 storage, bool mapped) noexcept -> self {
        auto buffer = self();
        glCreateBuffers(1, &buffer._id);
        glNamedBufferStorage(buffer._id, size, nullptr, storage);

        if (mapped) {
            buffer._mapped = glMapNamedBuffer(buffer._id, GL_READ_WRITE);
        }

        buffer._type = type;
        buffer._size = size;
        return buffer;
    }

    auto buffer_t::id() const noexcept -> uint32 {
        return _id;
    }

    auto buffer_t::size() const noexcept -> uint64 {
        return _size;
    }

    auto buffer_t::mapped() const noexcept -> void* {
        return _mapped;
    }

    auto buffer_t::write(const void* data, uint64 size, uint64 offset) const noexcept -> const self& {
        assert(offset + size <= _size && "overflow");
        if (size == 0) {
            return *this;
        }
        glNamedBufferSubData(_id, offset, size, data);
        return *this;
    }

    auto buffer_t::bind() const noexcept -> void {
        bind(_type);
    }

    auto buffer_t::bind_base(uint32 index) const noexcept -> const self& {
        return bind_base(_type, index);
    }

    auto buffer_t::bind_range(uint32 index, uint64 offset, uint64 size) const noexcept -> const self& {
        return bind_range(_type, index, offset, size);
    }

    auto buffer_t::bind(uint32 type) const noexcept -> void {
        glBindBuffer(type, _id);
    }

    auto buffer_t::bind_base(uint32 type, uint32 index) const noexcept -> const self& {
        glBindBufferBase(type, index, _id);
        return *this;
    }

    auto buffer_t::bind_range(uint32 type, uint32 index, uint64 offset, uint64 size) const noexcept -> const self& {
        if (size != 0) {
            glBindBufferRange(type, index, _id, offset, size);
        }
        return *this;
    }

    auto buffer_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
        swap(_type, other._type);
        swap(_size, other._size);
        swap(_mapped, other._mapped);
    }
} // namespace iris
