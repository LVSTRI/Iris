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

    auto texture_t::create(const fs::path& path, texture_type_t type) noexcept -> self {
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
            for (auto i = 0_i32; i < width * height * channels; i += channels) {
                if (data[i + 3] != 255) {
                    texture._is_opaque = false;
                    break;
                }
            }
        }

        glGenTextures(1, &texture._id);
        glBindTexture(GL_TEXTURE_2D, texture._id);

        if (!texture._is_opaque) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        const auto internal_format = type == texture_type_t::linear_srgb ? GL_RGBA8 : GL_SRGB8_ALPHA8;
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);

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

    auto texture_t::is_opaque() const noexcept -> bool {
        return _is_opaque;
    }

    auto texture_t::bind(uint32 index) const noexcept -> void {
        glActiveTexture(GL_TEXTURE0 + index);
        glBindTexture(GL_TEXTURE_2D, _id);
    }

    auto texture_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(other._id, _id);
        swap(other._width, _width);
        swap(other._height, _height);
        swap(other._channels, _channels);
        swap(other._is_opaque, _is_opaque);
    }
} // namespace iris
