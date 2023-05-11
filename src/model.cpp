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

    static auto is_texture_valid(const cgltf_texture* texture) noexcept {
        return texture && texture->basisu_image && texture->basisu_image->buffer_view && texture->basisu_image->buffer_view->buffer;
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

        auto options = cgltf_options();
        auto* gltf = (cgltf_data*)(nullptr);
        const auto s_path = path.generic_string();
        cgltf_parse_file(&options, s_path.c_str(), &gltf);
        cgltf_load_buffers(&options, gltf, s_path.c_str());

        auto texture_cache = std::unordered_map<const void*, uint32>();
        const auto import_texture = [&](const cgltf_texture* texture, texture_type_t type) {
            if (is_texture_valid(texture)) {
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
                    const auto* position_ptr = (glm::vec3*)(nullptr);
                    const auto* normal_ptr = (glm::vec3*)(nullptr);
                    const auto* uv_ptr = (glm::vec2*)(nullptr);
                    const auto* tangent_ptr = (glm::vec4*)(nullptr);

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
                    auto sphere = glm::vec4();
                    auto aabb = aabb_t();
                    aabb.min = glm::vec3(std::numeric_limits<float32>::max());
                    aabb.max = glm::vec3(std::numeric_limits<float32>::lowest());
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

                        aabb.min = glm::min(aabb.min, vertices[l].position);
                        aabb.max = glm::max(aabb.max, vertices[l].position);
                        sphere += glm::vec4(vertices[l].position, 0.0f);
                    }
                    aabb.center = (aabb.min + aabb.max) / 2.0f;
                    aabb.extent = aabb.max - aabb.center;
                    sphere /= static_cast<float32>(vertex_count);
                    //sphere = glm::make_vec4(aabb.center);

                    for (auto& vertex : vertices) {
                        sphere.w = glm::max(sphere.w, glm::distance(aabb.center, vertex.position));
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

                    if (is_texture_valid(diffuse_texture)) {
                        const auto& image = *diffuse_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        if (texture_cache.contains(ptr)) {
                            diffuse_texture_index = texture_cache.at(ptr);
                        }
                    }
                    if (is_texture_valid(normal_texture)) {
                        const auto& image = *normal_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        if (texture_cache.contains(ptr)) {
                            normal_texture_index = texture_cache.at(ptr);
                        }
                    }
                    if (is_texture_valid(specular_texture)) {
                        const auto& image = *specular_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        if (texture_cache.contains(ptr)) {
                            specular_texture_index = texture_cache.at(ptr);
                        }
                    }

                    auto m_mesh = mesh_pool.make_mesh(vertices, indices, vertex_format_as_attributes());
                    auto& object = model._objects.emplace_back();
                    object.scale = glm::vec3(1.0f);
                    if (node.has_scale) {
                        object.scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
                    }
                    object.mesh = std::move(m_mesh);
                    object.aabb = aabb;
                    object.sphere = sphere;
                    object.diffuse_texture = diffuse_texture_index;
                    object.normal_texture = normal_texture_index;
                    object.specular_texture = specular_texture_index;
                    cgltf_node_transform_world(&node, glm::value_ptr(model._transforms.emplace_back(glm::identity<glm::mat4>())));
                }

                for (auto j = 0_u32; j < node.children_count; ++j) {
                    nodes.push(node.children[j]);
                }
            }
        }

        iris::log("loaded model: \"", s_path, "\" has ", model._objects.size(), " objects and ", model._textures.size(), " textures");
        return model;
    }

    auto model_t::objects() const noexcept -> std::span<const object_t> {
        return _objects;
    }

    auto model_t::transforms() const noexcept -> std::span<const glm::mat4> {
        return _transforms;
    }

    auto model_t::textures() const noexcept -> std::span<const texture_t> {
        return _textures;
    }

    auto model_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_objects, other._objects);
        swap(_transforms, other._transforms);
        swap(_textures, other._textures);
    }

    meshlet_model_t::meshlet_model_t() noexcept = default;

    meshlet_model_t::~meshlet_model_t() noexcept = default;

    meshlet_model_t::meshlet_model_t(self&& other) noexcept {
        swap(other);
    }

    auto meshlet_model_t::operator =(self&& other) noexcept -> self& {
        self(std::move(other)).swap(*this);
        return *this;
    }

    auto meshlet_model_t::create(const fs::path& path) noexcept -> self {
        auto meshlet_model = self();
        auto options = cgltf_options();
        auto* gltf = (cgltf_data*)(nullptr);
        const auto s_path = path.generic_string();
        cgltf_parse_file(&options, s_path.c_str(), &gltf);
        cgltf_load_buffers(&options, gltf, s_path.c_str());

        auto texture_cache = std::unordered_map<const void*, uint32>();
        const auto import_texture = [&](const cgltf_texture* texture, texture_type_t type) {
            if (is_texture_valid(texture)) {
                const auto& image = *texture->basisu_image;
                const auto& buffer_view = *image.buffer_view;
                const auto& buffer = *buffer_view.buffer;
                const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                if (!texture_cache.contains(ptr)) {
                    texture_cache[ptr] = meshlet_model._textures.size();
                    meshlet_model._textures.emplace_back(texture_t::create_compressed(std::span(ptr, buffer.size), type));
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

        auto total_meshlets = 0_u32;
        auto vertex_offset = 0_u32;
        auto index_offset = 0_u32;
        auto triangle_offset = 0_u32;
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
                    const auto* position_ptr = (glm::vec3*)(nullptr);
                    const auto* normal_ptr = (glm::vec3*)(nullptr);
                    const auto* uv_ptr = (glm::vec2*)(nullptr);
                    const auto* tangent_ptr = (glm::vec4*)(nullptr);

                    auto vertices = std::vector<meshlet_vertex_format_t>();
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
                    auto sphere = glm::vec4();
                    auto aabb = aabb_t();
                    aabb.min = glm::vec3(std::numeric_limits<float>::max());
                    aabb.max = glm::vec3(std::numeric_limits<float>::lowest());
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

                        aabb.min = glm::min(aabb.min, vertices[l].position);
                        aabb.max = glm::max(aabb.max, vertices[l].position);
                    }
                    for (auto& vertex : vertices) {
                        sphere.w = glm::max(sphere.w, glm::length(vertex.position - aabb.center));
                    }
                    aabb.center = (aabb.min + aabb.max) / 2.0f;
                    aabb.extent = aabb.max - aabb.center;
                    sphere = glm::vec4(aabb.center, sphere.w);
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

                    constexpr auto max_vertices = 32u;
                    constexpr auto max_triangles = 124u;
                    constexpr auto cone_weight = 0.0f;
                    const auto max_meshlets = meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles);
                    auto meshlets = std::vector<meshopt_Meshlet>(max_meshlets);
                    auto meshlet_vertices = std::vector<uint32>(max_meshlets * max_vertices);
                    auto meshlet_triangles = std::vector<uint8>(max_meshlets * max_triangles * 3);
                    const auto meshlet_count = meshopt_buildMeshlets(
                        meshlets.data(),
                        meshlet_vertices.data(),
                        meshlet_triangles.data(),
                        indices.data(),
                        indices.size(),
                        (const float32*)vertices.data(),
                        vertices.size(),
                        sizeof(meshlet_vertex_format_t),
                        max_vertices,
                        max_triangles,
                        cone_weight);

                    auto& last_meshlet = meshlets[meshlet_count - 1];
                    meshlet_vertices.resize(last_meshlet.vertex_offset + last_meshlet.vertex_count);
                    meshlet_triangles.resize(last_meshlet.triangle_offset + ((last_meshlet.triangle_count * 3 + 3) & ~3));
                    meshlets.resize(meshlet_count);

                    meshlet_model._vertices.insert_range(meshlet_model._vertices.end(), vertices);
                    meshlet_model._indices.insert_range(meshlet_model._indices.end(), meshlet_vertices);
                    meshlet_model._triangles.insert_range(meshlet_model._triangles.end(), meshlet_triangles);

                    auto& meshlet_group = meshlet_model._meshlet_groups.emplace_back();
                    {
                        meshlet_group.vertex_count = vertices.size();
                        meshlet_group.vertex_offset = vertex_offset;
                        meshlet_group.meshlets.reserve(meshlet_count);

                        for (auto k = 0_u32; k < meshlet_count; ++k) {
                            auto& meshlet = meshlet_group.meshlets.emplace_back();
                            meshlet.vertex_offset = vertex_offset;
                            meshlet.index_offset = index_offset + meshlets[k].vertex_offset;
                            meshlet.index_count = meshlets[k].vertex_count;
                            meshlet.triangle_offset = triangle_offset + meshlets[k].triangle_offset;
                            meshlet.triangle_count = meshlets[k].triangle_count;
                        }
                    }

                    total_meshlets += meshlet_count;
                    vertex_offset += vertices.size();
                    index_offset += meshlet_vertices.size();
                    triangle_offset += meshlet_triangles.size();

                    auto diffuse_texture_index = -1_u32;
                    auto normal_texture_index = -1_u32;
                    auto specular_texture_index = -1_u32;

                    const auto& material = *primitive.material;
                    const auto* diffuse_texture = material.pbr_metallic_roughness.base_color_texture.texture;
                    const auto* normal_texture = material.normal_texture.texture;
                    const auto* specular_texture = material.pbr_specular_glossiness.specular_glossiness_texture.texture;

                    if (is_texture_valid(diffuse_texture)) {
                        const auto& image = *diffuse_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        if (texture_cache.contains(ptr)) {
                            diffuse_texture_index = texture_cache.at(ptr);
                        }
                    }
                    if (is_texture_valid(normal_texture)) {
                        const auto& image = *normal_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        if (texture_cache.contains(ptr)) {
                            normal_texture_index = texture_cache.at(ptr);
                        }
                    }
                    if (is_texture_valid(specular_texture)) {
                        const auto& image = *specular_texture->basisu_image;
                        const auto& buffer_view = *image.buffer_view;
                        const auto& buffer = *buffer_view.buffer;
                        const auto* ptr = static_cast<const uint8*>(buffer.data) + buffer_view.offset;
                        if (texture_cache.contains(ptr)) {
                            specular_texture_index = texture_cache.at(ptr);
                        }
                    }

                    meshlet_group.diffuse_index = diffuse_texture_index;
                    meshlet_group.normal_index = normal_texture_index;
                    meshlet_group.specular_index = specular_texture_index;
                    cgltf_node_transform_world(&node, glm::value_ptr(meshlet_model._transforms.emplace_back(glm::identity<glm::mat4>())));
                }

                for (auto j = 0_u32; j < node.children_count; ++j) {
                    nodes.push(node.children[j]);
                }
            }
        }
        meshlet_model._meshlet_count = total_meshlets;

        std::cout
            << "model: " << path << " has:\n"
            << "- " << meshlet_model._meshlet_groups.size() << " meshlet groups\n"
            << "- " << total_meshlets << " meshlets\n";
        return meshlet_model;
    }

    auto meshlet_model_t::meshlet_groups() const noexcept -> std::span<const meshlet_group_t> {
        return _meshlet_groups;
    }

    auto meshlet_model_t::transforms() const noexcept -> std::span<const glm::mat4> {
        return _transforms;
    }

    auto meshlet_model_t::textures() const noexcept -> std::span<const texture_t> {
        return _textures;
    }

    auto meshlet_model_t::vertices() const noexcept -> std::span<const meshlet_vertex_format_t> {
        return _vertices;
    }

    auto meshlet_model_t::indices() const noexcept -> std::span<const uint32> {
        return _indices;
    }

    auto meshlet_model_t::triangles() const noexcept -> std::span<const uint8> {
        return _triangles;
    }

    auto meshlet_model_t::meshlet_count() const noexcept -> uint32 {
        return _meshlet_count;
    }

    auto meshlet_model_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_transforms, other._transforms);
        swap(_textures, other._textures);
    }
} // namespace iris
