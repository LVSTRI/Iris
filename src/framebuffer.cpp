#include <framebuffer.hpp>

#include <glad/gl.h>

#include <array>

namespace iris {
    framebuffer_attachment_t::framebuffer_attachment_t() noexcept = default;

    framebuffer_attachment_t::~framebuffer_attachment_t() noexcept {
        glDeleteTextures(1, &_id);
    }

    framebuffer_attachment_t::framebuffer_attachment_t(self&& other) noexcept {
        swap(other);
    }

    auto framebuffer_attachment_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto framebuffer_attachment_t::create(uint32 width, uint32 height, int32 format, int32 base_format, uint32 type) noexcept -> self {
        auto attachment = self();
        glGenTextures(1, &attachment._id);
        glBindTexture(GL_TEXTURE_2D, attachment._id);

        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, base_format, type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const auto color = std::to_array({ 1.0f, 1.0f, 1.0f, 1.0f });
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color.data());

        attachment._width = width;
        attachment._height = height;
        attachment._format = format;
        attachment._base_format = base_format;
        attachment._type = type;
        return attachment;
    }

    auto framebuffer_attachment_t::id() const noexcept -> uint32 {
        return _id;
    }

    auto framebuffer_attachment_t::width() const noexcept -> uint32 {
        return _width;
    }

    auto framebuffer_attachment_t::height() const noexcept -> uint32 {
        return _height;
    }

    auto framebuffer_attachment_t::format() const noexcept -> uint32 {
        return _format;
    }

    auto framebuffer_attachment_t::base_format() const noexcept -> uint32 {
        return _base_format;
    }

    auto framebuffer_attachment_t::type() const noexcept -> uint32 {
        return _type;
    }

    auto framebuffer_attachment_t::bind() const noexcept -> void {
        glBindTexture(GL_TEXTURE_2D, _id);
    }

    auto framebuffer_attachment_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
        swap(_width, other._width);
        swap(_height, other._height);
        swap(_format, other._format);
        swap(_base_format, other._base_format);
        swap(_type, other._type);
    }

    framebuffer_t::framebuffer_t() noexcept = default;

    framebuffer_t::~framebuffer_t() noexcept {
        _attachments.clear();
        glDeleteFramebuffers(1, &_id);
    }

    framebuffer_t::framebuffer_t(self&& other) noexcept {
        swap(other);
    }

    auto framebuffer_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto framebuffer_t::create(std::vector<std::reference_wrapper<const framebuffer_attachment_t>> attachments) noexcept -> self {
        auto framebuffer = self();
        glGenFramebuffers(1, &framebuffer._id);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer._id);

        for (auto i = 0; i < attachments.size(); ++i) {
            const auto& u_attachment = attachments[i].get();
            auto target = 0_u32;
            switch (u_attachment.base_format()) {
                case GL_DEPTH_COMPONENT:
                    target = GL_DEPTH_ATTACHMENT;
                    break;

                case GL_STENCIL_INDEX:
                    target = GL_STENCIL_ATTACHMENT;
                    break;

                case GL_DEPTH_STENCIL:
                    target = GL_DEPTH_STENCIL_ATTACHMENT;
                    break;

                default:
                    target = GL_COLOR_ATTACHMENT0 + i;
                    break;
            }
            glFramebufferTexture2D(GL_FRAMEBUFFER, target, GL_TEXTURE_2D, u_attachment.id(), 0);
        }
        framebuffer._width = attachments[0].get().width();
        framebuffer._height = attachments[0].get().height();
        framebuffer._attachments = std::move(attachments);

        return framebuffer;
    }

    auto framebuffer_t::id() const noexcept -> uint32 {
        return _id;
    }

    auto framebuffer_t::width() const noexcept -> uint32 {
        return _width;
    }

    auto framebuffer_t::height() const noexcept -> uint32 {
        return _height;
    }

    auto framebuffer_t::attachments() const noexcept -> std::span<const std::reference_wrapper<const framebuffer_attachment_t>> {
        return _attachments;
    }

    auto framebuffer_t::bind() const noexcept -> void {
        glBindFramebuffer(GL_FRAMEBUFFER, _id);
    }

    auto framebuffer_t::attachment(uint32 index) -> const framebuffer_attachment_t& {
        return _attachments[index].get();
    }

    auto framebuffer_t::is_complete() const noexcept -> bool {
        bind();
        return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    }

    auto framebuffer_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
        swap(_width, other._width);
        swap(_height, other._height);
        swap(_attachments, other._attachments);
    }
} // namespace iris
