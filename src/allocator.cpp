#include <allocator.hpp>

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

    auto buffer_slice_t::create(uint64 offset, uint64 size, uint64 index, allocator_t* allocator) noexcept -> self {
        auto slice = self();
        slice._offset = offset;
        slice._size = size;
        slice._index = index;
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

    auto buffer_slice_t::allocator() const noexcept -> allocator_t& {
        return *_allocator;
    }

    auto buffer_slice_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_offset, other._offset);
        swap(_size, other._size);
        swap(_index, other._index);
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

    auto allocator_t::create() noexcept -> self {
        auto allocator = self();
        allocator._blocks.emplace_back().insert({ 0, capacity });
        return allocator;
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

    auto allocator_t::free(const buffer_slice_t& slice) noexcept -> void {
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
            block.insert(new_block);
        }
    }

    auto allocator_t::is_block_empty(uint64 block) const noexcept -> bool {
        auto page = *_blocks[block].begin();
        return page.size == capacity;
    }

    auto allocator_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_blocks, other._blocks);
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
            _blocks.emplace_back().insert({ 0, capacity });
            best = _blocks.back().begin();
            id = _blocks.size();
        }
        return std::make_pair(best, id - 1);
    }
} // namespace iris
