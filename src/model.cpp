#include <texture.hpp>
#include <model.hpp>
#include <mesh.hpp>

#include <glad/gl.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <stack>

namespace iris {
    static auto import_texture(const aiScene& scene,
                               const aiMaterial& material,
                               aiTextureType type,
                               const fs::path& root,
                               std::unordered_map<fs::path, texture_t>& textures) noexcept -> const texture_t* {
        if (material.GetTextureCount(type) > 0) {
            auto t_path = aiString();
            if (material.GetTexture(type, 0, &t_path) == AI_SUCCESS) {
                auto path = root / t_path.C_Str();
                if (path.string().contains('*')) {
                    const auto& e_texture = *scene.GetEmbeddedTexture(t_path.C_Str());
                    path = root / e_texture.mFilename.C_Str();
                }
                if (!path.has_extension()) {
                    path.replace_extension(".png");
                    if (!fs::exists(path)) {
                        path.replace_extension(".jpg");
                    }
                }
                if (!textures.contains(path)) {
                    const auto t_type = type == aiTextureType_DIFFUSE ?
                        texture_type_t::non_linear_srgb :
                        texture_type_t::linear_srgb;
                    textures[path] = texture_t::create(path, t_type);
                }
                return &textures[path];
            }
        }
        return nullptr;
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

    auto model_t::create(const fs::path& path) noexcept -> self {
        auto model = self();
        auto importer = Assimp::Importer();
        const auto importer_flags =
            aiProcess_Triangulate |
            aiProcess_FlipUVs |
            aiProcess_GenNormals;
        auto* scene = importer.ReadFile(path.string(), importer_flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            iris::log("assimp error: ", importer.GetErrorString());
            assert(false && "failed to load model");
        }

        const auto root = path.parent_path();
        auto nodes = std::vector<const aiNode*>();
        nodes.reserve(16);
        {
            auto stack = std::stack<const aiNode*>();
            stack.push(scene->mRootNode);
            while (!stack.empty()) {
                const auto* node = stack.top();
                stack.pop();
                nodes.push_back(node);
                for (auto i = 0_u32; i < node->mNumChildren; ++i) {
                    stack.push(node->mChildren[i]);
                }
            }
        }

        model._meshes.reserve(nodes.size());
        for (const auto* node : nodes) {
            for (auto i = 0_u32; i < node->mNumMeshes; ++i) {
                const auto& mesh = *scene->mMeshes[node->mMeshes[i]];
                // vertices processing
                auto vertices = std::vector<vertex_t>();
                vertices.reserve(mesh.mNumVertices);
                for (auto j = 0_u32; j < mesh.mNumVertices; ++j) {
                    auto& vertex = vertices.emplace_back();
                    vertex = {
                        .position = {
                            mesh.mVertices[j].x,
                            mesh.mVertices[j].y,
                            mesh.mVertices[j].z
                        },
                        .normal = {
                            mesh.mNormals[j].x,
                            mesh.mNormals[j].y,
                            mesh.mNormals[j].z
                        }
                    };
                    if (mesh.mTextureCoords[0]) {
                        vertex.uv = {
                            mesh.mTextureCoords[0][j].x,
                            mesh.mTextureCoords[0][j].y
                        };
                    }
                }
                // indices processing
                auto indices = std::vector<uint32>();
                indices.reserve(mesh.mNumFaces * 3);
                for (auto j = 0_u32; j < mesh.mNumFaces; ++j) {
                    const auto& face = mesh.mFaces[j];
                    for (auto k = 0_u32; k < face.mNumIndices; ++k) {
                        indices.emplace_back(face.mIndices[k]);
                    }
                }
                // texture processing
                auto textures = std::vector<std::reference_wrapper<const texture_t>>();
                if (mesh.mMaterialIndex >= 0) {
                    const auto& material = *scene->mMaterials[mesh.mMaterialIndex];
                    const auto* diffuse_texture = import_texture(*scene, material, aiTextureType_DIFFUSE, root, model._textures);
                    const auto* specular_texture = import_texture(*scene, material, aiTextureType_SPECULAR, root, model._textures);
                    if (diffuse_texture) {
                        textures.emplace_back(std::cref(*diffuse_texture));
                    }
                    if (specular_texture) {
                        textures.emplace_back(std::cref(*specular_texture));
                    }
                }
                // transform processing
                const auto* current = node;
                auto transform = glm::identity<glm::mat4>();
                while (current) {
                    transform = transform * glm::transpose(glm::make_mat4(current->mTransformation[0]));
                    current = current->mParent;
                }
                // insert mesh
                model._meshes.emplace_back(mesh_t::create(std::move(vertices), std::move(indices), std::move(textures), transform));
            }
        }
        iris::log("loaded model: \"", path.string(), "\" has ", model._meshes.size(), " meshes and ", model._textures.size(), " textures");
        return model;
    }

    auto model_t::meshes() const noexcept -> std::span<const mesh_t> {
        return _meshes;
    }

    auto model_t::swap(self& other) noexcept -> void {
        using std::swap;
        swap(_meshes, other._meshes);
        swap(_textures, other._textures);
    }
} // namespace iris
