#include <texture.hpp>

#include <glad/gl.h>

#include <ktx.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cassert>

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

    auto texture_t::create_compressed(std::span<const uint8> data, texture_type_t type, bool make_resident) noexcept -> self {
        auto texture = self();
        auto* ktx = (ktxTexture2*)(nullptr);
        auto result = ktxTexture2_CreateFromMemory(&data[0], data.size(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx);
        assert((result == KTX_SUCCESS) && "failed to load texture");

        auto width = ktx->baseWidth;
        auto height = ktx->baseHeight;
        auto channels = ktxTexture2_GetNumComponents(ktx);


        iris::log("loaded texture: \"", (const void*)&data[0], "\" (", width, "x", height, ")");
        texture._width = width;
        texture._height = height;
        texture._channels = channels;
        texture._is_opaque = channels <= 3;

        auto ktx_format = ktx_transcode_fmt_e();
        switch (type) {
            case texture_type_t::linear_r8g8_unorm:
                ktx_format = KTX_TTF_BC5_RG;
                break;

            case texture_type_t::linear_r8g8b8_unorm:
                ktx_format = KTX_TTF_BC1_RGB;
                break;

            case texture_type_t::linear_r8g8b8a8_unorm:
                ktx_format = KTX_TTF_BC3_RGBA;
                break;

            case texture_type_t::non_linear_r8g8b8a8_unorm:
                ktx_format = KTX_TTF_BC7_RGBA;
                break;
        }

        if (ktxTexture2_NeedsTranscoding(ktx)) {
            result = ktxTexture2_TranscodeBasis(ktx, ktx_format, KTX_TF_HIGH_QUALITY);
            assert((result == KTX_SUCCESS) && "failed to transcode texture");
        }

        auto format = 0_u32;
        switch (ktx_format) {
            case KTX_TTF_BC1_RGB:
                format = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
                break;

            case KTX_TTF_BC3_RGBA:
                format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
                break;

            case KTX_TTF_BC5_RG:
                format = GL_COMPRESSED_SIGNED_RG_RGTC2;
                break;

            case KTX_TTF_BC7_RGBA:
                format = GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM;
                break;

            default:
                break;
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

        glTextureStorage2D(texture._id, ktx->numLevels, format, width, height);
        for (auto level = 0_u32; level < ktx->numLevels; ++level) {
            auto offset = 0_u64;
            ktxTexture_GetImageOffset(ktxTexture(ktx), level, 0, 0, &offset);
            auto image_size = ktxTexture_GetImageSize(ktxTexture(ktx), level);
            const auto l_width = std::max(width >> level, 1_u32);
            const auto l_height = std::max(height >> level, 1_u32);
            glCompressedTextureSubImage2D(
                texture._id,
                level,
                0,
                0,
                l_width,
                l_height,
                format,
                image_size,
                ktx->pData + offset);
        }

        if (make_resident) {
            texture._handle = glGetTextureHandleARB(texture._id);
            glMakeTextureHandleResidentARB(texture._handle);
            texture._is_resident = true;
        }
        return texture;
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

        const auto internal_format = type != texture_type_t::non_linear_r8g8b8a8_unorm ? GL_RGBA8 : GL_SRGB8_ALPHA8;
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
