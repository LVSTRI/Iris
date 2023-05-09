#include <allocator.hpp>
#include <buffer.hpp>

#include <type_traits>
#include <utility>

namespace iris {
    buffer_slice_t::buffer_slice_t() noexcept = default;

    buffer_slice_t::~buffer_slice_t() noexcept {
        if (_allocator) {
            _allocator->free(*this);
        }
    }

    buffer_slice_t::buffer_slice_t(buffer_slice_t&& other) noexcept {
        swap(other);
    }

    auto buffer_slice_t::operator =(buffer_slice_t&& other) noexcept -> buffer_slice_t& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto buffer_slice_t::create(uint64 offset, uint64 size, uint64 index, allocator_t* allocator, buffer_t* buffer) noexcept -> self {
        auto slice = self();
        slice._offset = offset;
        slice._size = size;
        slice._index = index;
        slice._handle = buffer;
        slice._allocator = allocator;
        return slice;
    }

    auto buffer_slice_t::offset() const noexcept -> uint64 {
        return _offset;
    }

    auto buffer_slice_t::size() const noexcept -> uint64 {
        return _size;
    }

    auto buffer_slice_t::index() const noexcept -> uint64 {
        return _index;
    }

    auto buffer_slice_t::handle() const noexcept -> buffer_t& {
        return *_handle;
    }

    auto buffer_slice_t::allocator() const noexcept -> allocator_t& {
        return *_allocator;
    }

    auto buffer_slice_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_offset, other._offset);
        swap(_size, other._size);
        swap(_index, other._index);
        swap(_handle, other._handle);
        swap(_allocator, other._allocator);
    }

    allocator_t::allocator_t() noexcept = default;

    allocator_t::~allocator_t() noexcept = default;

    allocator_t::allocator_t(self&& other) noexcept {
        swap(other);
    }

    auto allocator_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto allocator_t::create(uint64 capacity) noexcept -> self {
        auto allocator = self();
        allocator._blocks.emplace_back().insert({ 0, capacity });
        allocator._capacity = capacity;
        return allocator;
    }

    auto allocator_t::capacity() const noexcept -> uint64 {
        return _capacity;
    }

    auto allocator_t::allocate(uint64 size) noexcept -> buffer_slice_t {
        auto [page, block] = _find_best(size);
        auto new_page = page_t {
            .offset = page->offset,
            .size = size,
            .id = block
        };
        _blocks[block].insert({
            .offset = new_page.offset + new_page.size,
            .size = page->size - new_page.size,
        });
        _blocks[block].erase(page);
        return buffer_slice_t::create(new_page.offset, new_page.size, new_page.id, this);
    }

    auto allocator_t::free(const buffer_slice_t& slice) noexcept -> bool {
        auto& block = _blocks[slice.index()];
        auto page = page_t {
            .offset = slice.offset(),
            .size = slice.size(),
        };
        auto [curr, _] = block.insert(page);
        // coalesce blocks
        auto prev = curr;
        if (prev != block.begin()) {
            prev = std::prev(prev, 1);
        }
        // prev -> curr coalesce
        if (prev != curr && prev->offset + prev->size == curr->offset) {
            auto new_block = page_t {
                .offset = prev->offset,
                .size = prev->size + curr->size,
            };
            block.erase(prev);
            block.erase(curr);
            auto _0 = false;
            std::tie(curr, _0) = block.insert(new_block);
        }

        // curr -> next coalesce
        auto next = curr;
        if (next != block.end()) {
            next = std::next(next, 1);
        }
        if (next != block.end() && curr->offset + curr->size == next->offset) {
            auto new_block = page_t {
                .offset = curr->offset,
                .size = curr->size + next->size,
            };
            block.erase(curr);
            block.erase(next);
            auto _0 = false;
            std::tie(curr, _0) = block.insert(new_block);
        }
        return curr->size == _capacity;
    }

    auto allocator_t::is_block_empty(uint64 block) const noexcept -> bool {
        auto page = *_blocks[block].begin();
        return page.size == _capacity;
    }

    auto allocator_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_blocks, other._blocks);
        swap(_capacity, other._capacity);
    }

    auto allocator_t::_find_best(uint64 size) noexcept -> std::pair<structure_type::iterator, uint64> {
        auto best = structure_type::iterator();
        auto id = 0_u32;
        auto success = false;
        for (; id < _blocks.size() && !success; ++id) {
            best = _blocks[id].begin();
            for (auto it = _blocks[id].begin(); it != _blocks[id].end(); ++it) {
                if (it->size >= size && it->size <= best->size) {
                    success = true;
                    best = it;
                } else if (it->size == size) {
                    success = true;
                    break;
                }
            }
        }

        // need to allocate more
        if (!success) {
            _blocks.emplace_back().insert({ 0, _capacity });
            best = _blocks.back().begin();
            id = _blocks.size();
        }
        return std::make_pair(best, id - 1);
    }

    buffer_allocator_t::buffer_allocator_t() noexcept = default;

    buffer_allocator_t::~buffer_allocator_t() noexcept = default;

    buffer_allocator_t::buffer_allocator_t(self&& other) noexcept {
        swap(other);
    }

    auto buffer_allocator_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto buffer_allocator_t::create(uint64 capacity) noexcept -> self {
        auto buffer_allocator = self();
        buffer_allocator._allocator = allocator_t::create(capacity);
        return buffer_allocator;
    }

    auto buffer_allocator_t::capacity() const noexcept -> uint64 {
        return _allocator.capacity();
    }

    auto buffer_allocator_t::allocate(uint64 size) noexcept -> buffer_slice_t {
        auto allocation = _allocator.allocate(size);
        if (allocation.index() >= _blocks.size()) {
            _blocks.resize(allocation.index() + 1);
        }
        if (!_blocks[allocation.index()].id()) {
            _blocks[allocation.index()] = buffer_t::create(capacity(), GL_ARRAY_BUFFER, GL_NONE);
        }
        return buffer_slice_t::create(
            allocation.offset(),
            allocation.size(),
            allocation.index(),
            &_allocator,
            &_blocks[allocation.index()]);
    }

    auto buffer_allocator_t::free(const buffer_slice_t& block) noexcept -> bool {
        if (_allocator.free(block)) {
            if (block.index() > 0) {
                _blocks[block.index()] = buffer_t();
                return true;
            }
        }
        return false;
    }

    auto buffer_allocator_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_allocator, other._allocator);
        swap(_blocks, other._blocks);
    }
} // namespace iris
