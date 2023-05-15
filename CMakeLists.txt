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

# cgltf setup
add_library(cgltf STATIC
    deps/cgltf/cgltf.h
    src/deps/cgltf.c)
target_include_directories(cgltf PUBLIC deps/cgltf)

# meshoptimizer setup
add_subdirectory(deps/meshoptimizer)

# KTX-Software setup
set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_VULKAN OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_GL_UPLOAD OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_WRITE OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/KTX-Software)

set_property(TARGET ktx PROPERTY CXX_STANDARD 17)

# imgui setup
add_library(imgui STATIC
    deps/imgui/imconfig.h
    deps/imgui/imgui.h
    deps/imgui/imgui.cpp
    deps/imgui/imgui_demo.cpp
    deps/imgui/imgui_draw.cpp
    deps/imgui/imgui_internal.h
    deps/imgui/imgui_tables.cpp
    deps/imgui/imgui_widgets.cpp
    deps/imgui/imstb_rectpack.h
    deps/imgui/imstb_textedit.h
    deps/imgui/imstb_truetype.h

    deps/imgui/backends/imgui_impl_glfw.h
    deps/imgui/backends/imgui_impl_glfw.cpp
    deps/imgui/backends/imgui_impl_opengl3.h
    deps/imgui/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC
    deps/imgui
    deps/imgui/backends)
target_link_libraries(imgui PUBLIC glad glfw)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(ktx PRIVATE /w)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ktx PRIVATE -march=native -Wno-everything)
endif()

add_library(Iris STATIC
    src/utilities.hpp
    src/shader.hpp
    src/shader.cpp
    src/texture.hpp
    src/texture.cpp
    src/camera.hpp
    src/camera.cpp
    src/model.hpp
    src/model.cpp
    src/framebuffer.hpp
    src/framebuffer.cpp
    src/debug_break.hpp
    src/buffer.hpp
    src/buffer.cpp
    src/allocator.hpp
    src/allocator.cpp
    src/mesh_pool.cpp
    src/mesh_pool.hpp)

target_compile_definitions(Iris PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(Iris PUBLIC glad glfw glm cgltf stb meshoptimizer ktx imgui)
target_include_directories(Iris PUBLIC src)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    target_compile_options(Iris PRIVATE -march=native -mavx2 -Wall -Wextra -Wpedantic -Wno-language-extension-token)
endif()

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

add_executable(IndirectDrawing src/5.0/main.cpp)
target_link_libraries(IndirectDrawing PUBLIC Iris)

add_executable(MeshShading src/5.1/main.cpp)
target_link_libraries(MeshShading PUBLIC Iris)

add_executable(AntiAliasing src/5.2/main.cpp)
target_link_libraries(AntiAliasing PUBLIC Iris)