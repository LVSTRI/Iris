#pragma once

#include <filesystem>
#include <functional>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>

namespace iris {
    namespace fs = std::filesystem;

    using int8 = std::int8_t;
    using int16 = std::int16_t;
    using int32 = std::int32_t;
    using int64 = std::int64_t;
    using uint8 = std::uint8_t;
    using uint16 = std::uint16_t;
    using uint32 = std::uint32_t;
    using uint64 = std::uint64_t;
    using float32 = float;
    using float64 = double;

    template <typename F>
    struct defer_t {
        constexpr defer_t(F&& func) noexcept
            : _func(std::forward<F>(func)) {}

        constexpr ~defer_t() noexcept {
            std::invoke(_func);
        }

        F _func = {};
    };

    template <typename... Args>
    auto log(Args&&... args) noexcept -> void {
        (std::cout << ... << args) << '\n';
    }

    inline auto whole_file(const fs::path& path) noexcept -> std::string {
        auto file = std::ifstream(path, std::ios::ate);
        auto result = std::string(file.tellg(), '\0');
        file.seekg(0, std::ios::beg);
        file.read(result.data(), result.size());
        return result;
    }

    template <typename T>
    concept is_container = requires (T value) {
        typename T::value_type;
        { value.size() } -> std::convertible_to<uint64>;
    };

    template <typename C>
        requires is_container<C>
    constexpr auto size_bytes(const C& container) noexcept -> uint64 {
        return container.size() * sizeof(typename C::value_type);
    }

    template <typename T>
    constexpr auto size_bytes(const T&) noexcept -> uint64 {
        return sizeof(T);
    }

    template <typename T>
    constexpr auto as_const_ptr(const T& value) noexcept -> const T* {
        return static_cast<const T*>(std::addressof(const_cast<T&>(value)));
    }

    inline namespace literals {
        constexpr auto operator ""_i8(unsigned long long int value) noexcept -> int8 {
            return static_cast<int8>(value);
        }

        constexpr auto operator ""_i16(unsigned long long int value) noexcept -> int16 {
            return static_cast<int16>(value);
        }

        constexpr auto operator ""_i32(unsigned long long int value) noexcept -> int32 {
            return static_cast<int32>(value);
        }

        constexpr auto operator ""_i64(unsigned long long int value) noexcept -> int64 {
            return static_cast<int64>(value);
        }

        constexpr auto operator ""_u8(unsigned long long int value) noexcept -> uint8 {
            return static_cast<uint8>(value);
        }

        constexpr auto operator ""_u16(unsigned long long int value) noexcept -> uint16 {
            return static_cast<uint16>(value);
        }

        constexpr auto operator ""_u32(unsigned long long int value) noexcept -> uint32 {
            return static_cast<uint32>(value);
        }

        constexpr auto operator ""_u64(unsigned long long int value) noexcept -> uint64 {
            return static_cast<uint64>(value);
        }

        constexpr auto operator ""_f32(long double value) noexcept -> float32 {
            return static_cast<float32>(value);
        }

        constexpr auto operator ""_f64(long double value) noexcept -> float64 {
            return static_cast<float64>(value);
        }
    } // namespace iris::literals
} // namespace iris
