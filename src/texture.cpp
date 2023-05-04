#include <texture.hpp>

#include <glad/gl.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace iris {
    texture_t::texture_t() noexcept = default;

    texture_t::~texture_t() noexcept {
        glDeleteTextures(1, &_id);
    }

    texture_t::texture_t(self&& other) noexcept {
        swap(other);
    }

    auto texture_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto texture_t::create(const fs::path& path, texture_type_t type, bool make_resident) noexcept -> self {
        auto texture = self();
        auto width = 0_i32;
        auto height = 0_i32;
        auto channels = 4_i32;
        auto* data = stbi_load(path.string().c_str(), &width, &height, &channels, channels);
        assert(data && "failed to load texture");

        iris::log("loaded texture: \"", path.string(), "\" (", width, "x", height, ")");
        texture._width = width;
        texture._height = height;
        texture._channels = channels;

        if (channels == 4 && path.extension() == ".png") {
            for (auto i = 0_u32; i < width * height * channels; i += channels) {
                if (data[i + 3] != 255) {
                    texture._is_opaque = false;
                    break;
                }
            }
        }

        glCreateTextures(GL_TEXTURE_2D, 1, &texture._id);

        if (!texture._is_opaque) {
            glTextureParameteri(texture._id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTextureParameteri(texture._id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glTextureParameteri(texture._id, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTextureParameteri(texture._id, GL_TEXTURE_WRAP_T, GL_REPEAT);
        }
        glTextureParameteri(texture._id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(texture._id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameterf(texture._id, GL_TEXTURE_MAX_ANISOTROPY, 16.0f);

        const auto internal_format = type == texture_type_t::linear_srgb ? GL_RGBA8 : GL_SRGB8_ALPHA8;
        const auto level_count = std::floor(std::log2(std::max(width, height))) + 1;
        glTextureStorage2D(texture._id, level_count, internal_format, width, height);
        glTextureSubImage2D(texture._id, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateTextureMipmap(texture._id);
        stbi_image_free(data);

        if (make_resident) {
            texture._handle = glGetTextureHandleARB(texture._id);
            glMakeTextureHandleResidentARB(texture._handle);
            texture._is_resident = true;
        }
        return texture;
    }

    auto texture_t::id() const noexcept -> uint32 {
        return _id;
    }

    auto texture_t::width() const noexcept -> uint32 {
        return _width;
    }

    auto texture_t::height() const noexcept -> uint32 {
        return _height;
    }

    auto texture_t::channels() const noexcept -> uint32 {
        return _channels;
    }

    auto texture_t::handle() const noexcept -> uint64 {
        return _handle;
    }

    auto texture_t::is_opaque() const noexcept -> bool {
        return _is_opaque;
    }

    auto texture_t::bind(uint32 index) const noexcept -> void {
        glBindTextureUnit(index, _id);
    }

    auto texture_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(other._id, _id);
        swap(other._width, _width);
        swap(other._height, _height);
        swap(other._channels, _channels);
        swap(other._handle, _handle);
        swap(other._is_opaque, _is_opaque);
        swap(other._is_resident, _is_resident);
    }
} // namespace iris
