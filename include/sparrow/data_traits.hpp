

#pragma once

#include "data_type.hpp"
#include "fixed_size_layout.hpp"
#include "variable_size_binary_layout.hpp"

namespace sparrow
{

    template<class T>
    struct common_native_types_traits
    {
        using value_type = T;
        using default_layout = fixed_size_layout<T>;
    };

    template<>
    struct arrow_traits<std::nullopt_t>
    {
        static constexpr data_type type_id = data_type::NA;
        using value_type = std::nullopt_t;
        using default_layout = fixed_size_layout<value_type>; // TODO: replace this by a special layout that's always empty

    };

    template<>
    struct arrow_traits<bool> : common_native_types_traits<bool>
    {
        static constexpr data_type type_id = data_type::BOOL;
    };

    template<>
    struct arrow_traits<std::uint8_t> : common_native_types_traits<std::uint8_t>
    {
        static constexpr data_type type_id = data_type::UINT8;
    };

    template<>
    struct arrow_traits<std::int8_t> : common_native_types_traits<std::int8_t>
    {
        static constexpr data_type type_id = data_type::INT8;
    };

    template<>
    struct arrow_traits<std::uint16_t> : common_native_types_traits<std::uint16_t>
    {
        static constexpr data_type type_id = data_type::UINT16;
    };

    template<>
    struct arrow_traits<std::int16_t> : common_native_types_traits<std::int16_t>
    {
        static constexpr data_type type_id = data_type::INT16;
    };

    template<>
    struct arrow_traits<std::uint32_t> : common_native_types_traits<std::uint32_t>
    {
        static constexpr data_type type_id = data_type::UINT32;
    };

    template<>
    struct arrow_traits<std::int32_t> : common_native_types_traits<std::int32_t>
    {
        static constexpr data_type type_id = data_type::INT32;
    };

    template<>
    struct arrow_traits<std::uint64_t> : common_native_types_traits<std::uint64_t>
    {
        static constexpr data_type type_id = data_type::UINT64;
    };

    template<>
    struct arrow_traits<std::int64_t> : common_native_types_traits<std::int64_t>
    {
        static constexpr data_type type_id = data_type::INT64;
    };

    template<>
    struct arrow_traits<float16_t> : common_native_types_traits<float16_t>
    {
        static constexpr data_type type_id = data_type::HALF_FLOAT;
    };

    template<>
    struct arrow_traits<float32_t> : common_native_types_traits<float32_t>
    {
        static constexpr data_type type_id = data_type::FLOAT;
    };

    template<>
    struct arrow_traits<float64_t> : common_native_types_traits<float64_t>
    {
        static constexpr data_type type_id = data_type::DOUBLE;
    };

    template<>
    struct arrow_traits<std::string>
    {
        static constexpr data_type type_id = data_type::STRING;
        using value_type = std::string;
        using default_layout = variable_size_binary_layout<value_type, std::string_view, std::string_view>; // FIXME: this is incorrect, change when we have the right types

    };


}