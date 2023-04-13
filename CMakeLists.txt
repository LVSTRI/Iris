cmake_minimum_required(VERSION 3.21)
project(Iris)

set(CMAKE_CXX_STANDARD 23)

# GLAD setup
add_subdirectory(deps/glad/cmake)
glad_add_library(glad STATIC LANGUAGE c LOADER API gl:core=4.6)

# GLFW setup
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/glfw)

# STB setup
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE deps/stb)

# GLM setup
add_subdirectory(deps/glm)
target_compile_definitions(glm INTERFACE GLM_FORCE_AVX2)

# Assimp
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL_PDB OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_OBJ_IMPORTER ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_GLTF_IMPORTER ON CACHE BOOL "" FORCE)
add_subdirectory(deps/assimp)

add_library(Iris STATIC
    src/utilities.hpp
    src/shader.hpp
    src/shader.cpp
    src/texture.hpp
    src/texture.cpp
    src/camera.hpp
    src/camera.cpp
    src/mesh.hpp
    src/mesh.cpp
    src/model.hpp
    src/model.cpp
    src/framebuffer.hpp
    src/framebuffer.cpp
    src/debug_break.hpp
    src/buffer.hpp
    src/buffer.cpp)

target_link_libraries(Iris PUBLIC glad glfw stb glm assimp)
target_include_directories(Iris PUBLIC src)

add_executable(MousePicking src/2.1/main.cpp)
target_link_libraries(MousePicking PUBLIC Iris)

add_executable(Framebuffers src/3.1/main.cpp)
target_link_libraries(Framebuffers PUBLIC Iris)

add_executable(Buffers src/3.2/main.cpp)
target_link_libraries(Buffers PUBLIC Iris)

add_executable(AdvancedLighting src/4.1/main.cpp)
target_link_libraries(AdvancedLighting PUBLIC Iris)

add_executable(BasicRaytracing src/4.2/main.cpp)
target_link_libraries(BasicRaytracing PUBLIC Iris)
