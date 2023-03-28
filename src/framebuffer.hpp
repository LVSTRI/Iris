#pragma once

#include <utilities.hpp>

#include <vector>
#include <vector>
#include <span>

namespace iris {
    class framebuffer_attachment_t {
    public:
        using self = framebuffer_attachment_t;

        framebuffer_attachment_t() noexcept;
        ~framebuffer_attachment_t() noexcept;

        framebuffer_attachment_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        framebuffer_attachment_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(uint32 width, uint32 height, int32 format, int32 base_format, uint32 type) noexcept -> self;

        auto id() const noexcept -> uint32;
        auto width() const noexcept -> uint32;
        auto height() const noexcept -> uint32;
        auto format() const noexcept -> uint32;
        auto base_format() const noexcept -> uint32;
        auto type() const noexcept -> uint32;

        auto bind() const noexcept -> void;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _id = 0;
        uint32 _width = 0;
        uint32 _height = 0;
        int32 _format = 0;
        int32 _base_format = 0;
        uint32 _type = 0;
    };

    class framebuffer_t {
    public:
        using self = framebuffer_t;

        framebuffer_t() noexcept;
        ~framebuffer_t() noexcept;

        framebuffer_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        framebuffer_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(std::vector<std::reference_wrapper<const framebuffer_attachment_t>> attachments) noexcept -> self;

        auto id() const noexcept -> uint32;
        auto width() const noexcept -> uint32;
        auto height() const noexcept -> uint32;
        auto attachments() const noexcept -> std::span<const std::reference_wrapper<const framebuffer_attachment_t>>;

        auto bind() const noexcept -> void;

        auto attachment(uint32 index) -> const framebuffer_attachment_t&;
        auto is_complete() const noexcept -> bool;

        auto swap(self& other) noexcept -> void;

    private:
        uint32 _id = 0;
        uint32 _width = 0;
        uint32 _height = 0;

        std::vector<std::reference_wrapper<const framebuffer_attachment_t>> _attachments;
    };
} // namespace iris
