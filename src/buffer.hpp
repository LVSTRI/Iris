#pragma once

#include <utilities.hpp>

namespace iris {
    class buffer_t {
    public:
        using self = buffer_t;

        buffer_t() noexcept;
        ~buffer_t() noexcept;

        buffer_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        buffer_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(uint32 size, uint32 type, bool mapped = false) noexcept -> self;

        auto id() const noexcept -> uint32;
        auto size() const noexcept -> uint64;
        auto mapped() const noexcept -> void*;

        auto write(const void* data, uint64 size, uint64 offset = 0) const noexcept -> const self&;

        auto bind() const noexcept -> void;
        auto bind_base(uint32 index) const noexcept -> const self&;
        auto bind_range(uint32 index, uint64 offset, uint64 size) const noexcept -> const self&;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _id = 0;
        uint32 _type = 0;
        uint64 _size = 0;

        void* _mapped = nullptr;
    };
} // namespace iris
