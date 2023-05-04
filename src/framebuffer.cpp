#include <framebuffer.hpp>

#include <glad/gl.h>

#include <array>

namespace iris {
    static auto base_format_to_attachment(uint32 base_format, uint32 index = -1) noexcept -> uint32 {
        switch (base_format) {
            case GL_DEPTH_COMPONENT: return GL_DEPTH_ATTACHMENT;
            case GL_STENCIL_INDEX: return GL_STENCIL_ATTACHMENT;
            case GL_DEPTH_STENCIL: return GL_DEPTH_STENCIL_ATTACHMENT;
            default: return GL_COLOR_ATTACHMENT0 + index;
        }
    }

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

    auto framebuffer_attachment_t::create(
            uint32 width,
            uint32 height,
            uint32 layers,
            int32 format,
            int32 base_format,
            uint32 type,
            bool nearest) noexcept -> self {
        auto attachment = self();
        const auto target = layers == 1 ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY;

        glCreateTextures(target, 1, &attachment._id);

        if (layers > 1) {
            glTextureStorage3D(attachment._id, 1, format, width, height, layers);
        } else {
            glTextureStorage2D(attachment._id, 1, format, width, height);
        }
        if (nearest) {
            glTextureParameteri(attachment._id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(attachment._id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        } else {
            glTextureParameteri(attachment._id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(attachment._id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        glTextureParameteri(attachment._id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTextureParameteri(attachment._id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const auto color = std::to_array({ 1.0f, 1.0f, 1.0f, 1.0f });
        glTextureParameterfv(attachment._id, GL_TEXTURE_BORDER_COLOR, color.data());

        attachment._width = width;
        attachment._height = height;
        attachment._layers = layers;
        attachment._format = format;
        attachment._base_format = base_format;
        attachment._type = type;
        attachment._target = target;
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

    auto framebuffer_attachment_t::layers() const noexcept -> uint32 {
        return _layers;
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

    auto framebuffer_attachment_t::target() const noexcept -> uint32 {
        return _target;
    }

    auto framebuffer_attachment_t::bind() const noexcept -> void {
        glBindTexture(_target, _id);
    }

    auto framebuffer_attachment_t::bind_texture(uint32 index) const noexcept -> void {
        glBindTextureUnit(index, _id);
    }

    auto
    framebuffer_attachment_t::bind_image_texture(uint32 index, uint32 level, bool layered, uint32 layer, uint32 access) const noexcept
        -> void {
        glBindImageTexture(index, _id, level, layered, layer, access, _format);
    }

    auto framebuffer_attachment_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
        swap(_width, other._width);
        swap(_height, other._height);
        swap(_layers, other._layers);
        swap(_format, other._format);
        swap(_base_format, other._base_format);
        swap(_type, other._type);
        swap(_target, other._target);
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
        glCreateFramebuffers(1, &framebuffer._id);

        for (auto i = 0; i < attachments.size(); ++i) {
            const auto& u_attachment = attachments[i].get();
            auto target = base_format_to_attachment(u_attachment.base_format(), i);
            glNamedFramebufferTexture(framebuffer._id, target, u_attachment.id(), 0);
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

    auto framebuffer_t::clear_depth(float32 depth) const noexcept -> void {
        glClearNamedFramebufferfv(_id, GL_DEPTH, 0, &depth);
    }

    auto framebuffer_t::clear_depth_stencil(float32 depth, uint32 stencil) const noexcept -> void {
        glClearNamedFramebufferfi(_id, GL_DEPTH_STENCIL, 0, depth, stencil);
    }

    auto framebuffer_t::clear_color(uint32 index, const float32(&color)[]) const noexcept -> void {
        glClearNamedFramebufferfv(_id, GL_COLOR, index, color);
    }

    auto framebuffer_t::clear_color(uint32 index, const uint32(&color)[]) const noexcept -> void {
        glClearNamedFramebufferuiv(_id, GL_COLOR, index, color);
    }

    auto framebuffer_t::attachment(uint32 index) -> const framebuffer_attachment_t& {
        return _attachments[index].get();
    }

    auto framebuffer_t::is_complete() const noexcept -> bool {
        return glCheckNamedFramebufferStatus(_id, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    }

    auto framebuffer_t::set_layer(uint32 index, uint32 layer) const noexcept -> void {
        const auto& u_attachment = _attachments[index].get();
        if (u_attachment.layers() > 1) {
            glNamedFramebufferTextureLayer(
                _id,
                base_format_to_attachment(u_attachment.base_format()),
                u_attachment.id(),
                0,
                layer);
        }
    }

    auto framebuffer_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_id, other._id);
        swap(_width, other._width);
        swap(_height, other._height);
        swap(_attachments, other._attachments);
    }
} // namespace iris
