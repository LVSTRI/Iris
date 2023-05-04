#pragma once

#include <utilities.hpp>

#include <glm/mat4x4.hpp>

#include <unordered_map>
#include <filesystem>
#include <vector>
#include <span>

namespace iris {
    struct mesh_t;
    class mesh_pool_t;

    class model_t {
    public:
        using self = model_t;

        model_t() noexcept;
        ~model_t() noexcept;

        model_t(const self&) noexcept = delete;
        auto operator =(const self&) noexcept -> self& = delete;
        model_t(self&& other) noexcept;
        auto operator =(self&& other) noexcept -> self&;

        static auto create(mesh_pool_t& mesh_pool, const fs::path& path) noexcept -> self;

        auto meshes() const noexcept -> std::span<const mesh_t>;
        auto transforms() const noexcept -> std::span<const glm::mat4>;
        auto textures() const noexcept -> std::span<const texture_t>;

        auto swap(self& other) noexcept -> void;

    private:
        std::vector<mesh_t> _meshes;
        std::vector<glm::mat4> _transforms;
        std::vector<texture_t> _textures;
    };
} // namespace iris
