#pragma once

#include <utilities.hpp>

#include <vector>
#include <set>

namespace iris {
    class allocator_t;
    class buffer_slice_t {
    public:
        using self = buffer_slice_t;

        buffer_slice_t() noexcept;
        ~buffer_slice_t() noexcept;

        buffer_slice_t(const buffer_slice_t&) noexcept = delete;
        auto operator =(const buffer_slice_t&) noexcept -> buffer_slice_t& = delete;
        buffer_slice_t(buffer_slice_t&& other) noexcept;
        auto operator =(buffer_slice_t&& other) noexcept -> buffer_slice_t&;

        static auto create(uint64 offset, uint64 size, uint64 index, allocator_t* allocator) noexcept -> self;

        auto offset() const noexcept -> uint64;
        auto size() const noexcept -> uint64;
        auto index() const noexcept -> uint64;
        auto allocator() const noexcept -> allocator_t&;

        auto swap(self& other) noexcept -> void;

    private:
        uint64 _offset = 0;
        uint64 _size = 0;
        uint64 _index = 0;

        allocator_t* _allocator = nullptr;
    };

    class allocator_t {
    public:
        struct page_t {
            uint64 offset = 0;
            uint64 size = 0;
            // id of the block this page belongs to
            uint64 id = 0;

            constexpr auto operator <(const page_t& other) const noexcept -> bool {
                return offset < other.offset;
            }
        };

        using self = allocator_t;
        using structure_type = std::set<page_t>;

        allocator_t() noexcept;
        ~allocator_t() noexcept;

        allocator_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        allocator_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(uint64 capacity) noexcept -> self;

        auto capacity() const noexcept -> uint64;

        auto allocate(uint64 size) noexcept -> buffer_slice_t;
        auto free(const buffer_slice_t& block) noexcept -> void;

        auto is_block_empty(uint64 block) const noexcept -> bool;

        auto swap(self& other) noexcept -> void;

    private:
        auto _find_best(uint64 size) noexcept -> std::pair<structure_type::iterator, uint64>;

        std::vector<structure_type> _blocks;
        uint64 _capacity = 0;
    };
} // namespace iris
