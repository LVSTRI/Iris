#include <texture.hpp>
#include <model.hpp>
#include <mesh_pool.hpp>

#include <glad/gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cgltf.h>

#include <vector>
#include <queue>

namespace iris {
    struct vertex_format_t {
        glm::vec3 position = {};
        glm::vec3 normal = {};
        glm::vec2 uv = {};
        glm::vec4 tangent = {};
    };

    static auto vertex_format_as_attributes() noexcept {
        return std::vector<vertex_attribute_t>{
            { 0, 3 },
            { 1, 3 },
            { 2, 2 },
            { 3, 4 },
        };
    }

    static auto decode_texture_path(const fs::path& base, const cgltf_image* image) noexcept {
        auto path = fs::path();
        if (!image->uri) {
            path = base / image->name;
            if (!path.has_extension()) {
                if (std::strcmp(image->mime_type, "image/png") == 0) {
                    path.replace_extension(".png");
                } else if (std::strcmp(image->mime_type, "image/jpeg") == 0) {
                    path.replace_extension(".jpg");
                }
            }
        } else {
            path = base / image->uri;
        }
        return path.generic_string();
    }

    model_t::model_t() noexcept = default;

    model_t::~model_t() noexcept = default;

    model_t::model_t(self&& other) noexcept {
        swap(other);
    }

    auto model_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    // TODO: temporary, "model_t" should be a simple container, it should NOT upload things to the GPU nor invoke "mesh_pool_t"
    auto model_t::create(mesh_pool_t& mesh_pool, const fs::path& path) noexcept -> self {
        auto model = self();
        auto opt_path = path;
        if (!path.generic_string().contains("compressed")) {
            auto root_path = path
                .parent_path()
                .parent_path();
            opt_path =
                root_path /
                "compressed" /
                path.stem() /
                path.filename().replace_extension(".glb");
        }

        assert(fs::exists(opt_path) && "glTF mdel was not optimized");

        auto options = cgltf_options();
        auto* gltf = std::type_identity_t<cgltf_data*>();
        const auto s_path = opt_path.generic_string();
        cgltf_parse_file(&options, s_path.c_str(), &gltf);
        cgltf_load_buffers(&options, gltf, s_path.c_str());

        auto texture_cache = std::unordered_map<const void*, uint32>();
        const auto import_texture = [&](const cgltf_texture* texture, texture_type_t type) {
            if (texture) {
                const auto& image = *texture->basisu_image;
                const auto& buffer_view = *image.buffer_view;
                const auto& buffer = *buffer_view.buffer;
                const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                if (!texture_cache.contains(ptr)) {
                    texture_cache[ptr] = model._textures.size();
                    model._textures.emplace_back(texture_t::create_compressed(std::span(ptr, buffer.size), type));
                }
            }
        };
        for (auto i = 0_u32; i < gltf->materials_count; ++i) {
            const auto& material = gltf->materials[i];
            const auto* texture = material.pbr_metallic_roughness.base_color_texture.texture;
            if (material.has_pbr_metallic_roughness) {
                import_texture(texture, texture_type_t::non_linear_r8g8b8a8_unorm);
            }

            import_texture(material.normal_texture.texture, texture_type_t::linear_r8g8b8_unorm);

            texture = material.pbr_specular_glossiness.specular_glossiness_texture.texture;
            if (material.has_pbr_specular_glossiness) {
                import_texture(texture, texture_type_t::linear_r8g8b8_unorm);
            }
        }

        for (auto i = 0_u32; i < gltf->scene->nodes_count; ++i) {
            auto nodes = std::queue<const cgltf_node*>();
            nodes.push(gltf->scene->nodes[i]);
            while (!nodes.empty()) {
                const auto& node = *nodes.front();
                nodes.pop();
                if (!node.mesh) {
                    for (auto j = 0_u32; j < node.children_count; ++j) {
                        nodes.push(node.children[j]);
                    }
                    continue;
                }
                const auto& mesh = *node.mesh;
                for (auto j = 0_u32; j < mesh.primitives_count; ++j) {
                    const auto& primitive = mesh.primitives[j];
                    const auto* position_ptr = std::type_identity_t<glm::vec3*>();
                    const auto* normal_ptr = std::type_identity_t<glm::vec3*>();
                    const auto* uv_ptr = std::type_identity_t<glm::vec2*>();
                    const auto* tangent_ptr = std::type_identity_t<glm::vec4*>();

                    auto vertices = std::vector<vertex_format_t>();
                    auto vertex_count = 0_u32;
                    for (uint32 k = 0; k < primitive.attributes_count; ++k) {
                        const auto& attribute = primitive.attributes[k];
                        const auto& accessor = *attribute.data;
                        const auto& buffer_view = *accessor.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto& data_ptr = static_cast<const char*>(buffer.data);
                        switch (attribute.type) {
                            case cgltf_attribute_type_position:
                                vertex_count = accessor.count;
                                position_ptr = reinterpret_cast<const glm::vec3*>(data_ptr + buffer_view.offset + accessor.offset);
                                break;

                            case cgltf_attribute_type_normal:
                                normal_ptr = reinterpret_cast<const glm::vec3*>(data_ptr + buffer_view.offset + accessor.offset);
                                break;

                            case cgltf_attribute_type_texcoord:
                                if (!uv_ptr) {
                                    uv_ptr = reinterpret_cast<const glm::vec2*>(data_ptr + buffer_view.offset + accessor.offset);
                                }
                                break;

                            case cgltf_attribute_type_tangent:
                                tangent_ptr = reinterpret_cast<const glm::vec4*>(data_ptr + buffer_view.offset + accessor.offset);
                                break;

                            default: break;
                        }
                    }
                    vertices.resize(vertex_count);
                    for (auto l = 0_u32; l < vertex_count; ++l) {
                        std::memcpy(&vertices[l].position, position_ptr + l, sizeof(glm::vec3));
                        if (normal_ptr) {
                            std::memcpy(&vertices[l].normal, normal_ptr + l, sizeof(glm::vec3));
                        }
                        if (uv_ptr) {
                            std::memcpy(&vertices[l].uv, uv_ptr + l, sizeof(glm::vec2));
                        }
                        if (tangent_ptr) {
                            std::memcpy(&vertices[l].tangent, tangent_ptr + l, sizeof(glm::vec4));
                        }
                    }

                    auto indices = std::vector<uint32>();
                    {
                        const auto& accessor = *primitive.indices;
                        const auto& buffer_view = *accessor.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto& data_ptr = static_cast<const char*>(buffer.data);
                        indices.reserve(accessor.count);
                        switch (accessor.component_type) {
                            case cgltf_component_type_r_8:
                            case cgltf_component_type_r_8u: {
                                const auto* ptr = reinterpret_cast<const uint8*>(data_ptr + buffer_view.offset + accessor.offset);
                                std::ranges::copy(std::span(ptr, accessor.count), std::back_inserter(indices));
                            } break;

                            case cgltf_component_type_r_16:
                            case cgltf_component_type_r_16u: {
                                const auto* ptr = reinterpret_cast<const uint16*>(data_ptr + buffer_view.offset + accessor.offset);
                                std::ranges::copy(std::span(ptr, accessor.count), std::back_inserter(indices));
                            } break;

                            case cgltf_component_type_r_32f:
                            case cgltf_component_type_r_32u: {
                                const auto* ptr = reinterpret_cast<const uint32*>(data_ptr + buffer_view.offset + accessor.offset);
                                std::ranges::copy(std::span(ptr, accessor.count), std::back_inserter(indices));
                            } break;

                            default: break;
                        }
                    }

                    auto diffuse_texture_index = -1_u32;
                    auto normal_texture_index = -1_u32;
                    auto specular_texture_index = -1_u32;

                    const auto& material = *primitive.material;
                    const auto* diffuse_texture = material.pbr_metallic_roughness.base_color_texture.texture;
                    const auto* normal_texture = material.normal_texture.texture;
                    const auto* specular_texture = material.pbr_specular_glossiness.specular_glossiness_texture.texture;

                    if (diffuse_texture && diffuse_texture->basisu_image) {
                        const auto& image = *diffuse_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        diffuse_texture_index = texture_cache.at(ptr);
                    }
                    if (normal_texture && normal_texture->basisu_image) {
                        const auto& image = *normal_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        normal_texture_index = texture_cache.at(ptr);
                    }
                    if (specular_texture && specular_texture->basisu_image) {
                        const auto& image = *specular_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        specular_texture_index = texture_cache.at(ptr);
                    }

                    auto& m_mesh = model._meshes.emplace_back(mesh_pool.make_mesh(
                        vertices,
                        indices,
                        vertex_format_as_attributes()));
                    m_mesh.diffuse_texture = diffuse_texture_index;
                    m_mesh.normal_texture = normal_texture_index;
                    m_mesh.specular_texture = specular_texture_index;
                    cgltf_node_transform_world(&node, glm::value_ptr(model._transforms.emplace_back(glm::identity<glm::mat4>())));
                }

                for (auto j = 0_u32; j < node.children_count; ++j) {
                    nodes.push(node.children[j]);
                }
            }
        }

        iris::log("loaded model: \"", s_path, "\" has ", model._meshes.size(), " meshes and ", model._textures.size(), " textures");
        return model;
    }

    auto model_t::meshes() const noexcept -> std::span<const mesh_t> {
        return _meshes;
    }

    auto model_t::transforms() const noexcept -> std::span<const glm::mat4> {
        return _transforms;
    }

    auto model_t::textures() const noexcept -> std::span<const texture_t> {
        return _textures;
    }

    auto model_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_meshes, other._meshes);
        swap(_transforms, other._transforms);
        swap(_textures, other._textures);
    }
} // namespace iris
