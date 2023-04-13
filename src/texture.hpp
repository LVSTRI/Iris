#pragma once

#include <utilities.hpp>

namespace iris {
    class texture_t {
    public:
        using self = texture_t;

        texture_t() noexcept;
        ~texture_t() noexcept;

        texture_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        texture_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(const fs::path& path) noexcept -> self;

        auto id() const noexcept -> uint32;
        auto width() const noexcept -> uint32;
        auto height() const noexcept -> uint32;
        auto channels() const noexcept -> uint32;
        auto is_opaque() const noexcept -> bool;

        auto bind(uint32 index) const noexcept -> void;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _id = 0;
        uint32 _width = 0;
        uint32 _height = 0;
        uint32 _channels = 0;

        bool _is_opaque = true;
    };
} // namespace iris
