#pragma once

#include <utilities.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>
#include <span>

namespace iris {
    class mesh_t;
    class model_t {
    public:
        using self = model_t;

        model_t() noexcept;
        ~model_t() noexcept;

        model_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        model_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(const fs::path& path) noexcept -> self;

        auto meshes() const noexcept -> std::span<const mesh_t>;

        auto swap(self& other) noexcept -> void;

    private:
        std::vector<mesh_t> _meshes;
        std::unordered_map<fs::path, texture_t> _textures;
    };
} // namespace iris
