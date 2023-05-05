#pragma once

#include <utilities.hpp>

#include <span>

namespace iris {
    enum class texture_type_t : uint32 {
        linear_r8g8_unorm,
        linear_r8g8b8_unorm,
        linear_r8g8b8a8_unorm,
        non_linear_r8g8b8a8_unorm,
    };

    class texture_t {
    public:
        using self = texture_t;

        texture_t() noexcept;
        ~texture_t() noexcept;

        texture_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        texture_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create_compressed(std::span<const uint8> data, texture_type_t type, bool make_resident = true) noexcept -> self;
        static auto create(const fs::path& path, texture_type_t type, bool make_resident = true) noexcept -> self;

        auto id() const noexcept -> uint32;
        auto width() const noexcept -> uint32;
        auto height() const noexcept -> uint32;
        auto channels() const noexcept -> uint32;
        auto handle() const noexcept -> uint64;
        auto is_opaque() const noexcept -> bool;

        auto bind(uint32 index) const noexcept -> void;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _id = 0;
        uint32 _width = 0;
        uint32 _height = 0;
        uint32 _channels = 0;

        uint64 _handle = 0;

        bool _is_opaque = true;
        bool _is_resident = false;
    };
} // namespace iris
