// Copyright 2024 Man Group Operations Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

#include "sparrow/mp_utils.hpp"

// TODO: use exclusively `std::float16_t etc. once we switch to c++23, see https://en.cppreference.com/w/cpp/types/floating-point
#if __cplusplus <= 202002L
#   include "details/3rdparty/float16_t.hpp"
#else
#   include <stdfloat>
#endif


namespace sparrow
{
// TODO: use exclusively `std::float16_t etc. once we switch to c++23, see https://en.cppreference.com/w/cpp/types/floating-point
#if __cplusplus <= 202002L
    using float16_t = numeric::float16_t;
    using float32_t = float;
    using float64_t = double;
#else
    using float16_t = std::float16_t;
    using float32_t = std::float32_t;
    using float64_t = std::float64_t;
#endif

    // We need to be sure the current target platform is setup to support correctly these types.
    static_assert(sizeof(float16_t) == 2);
    static_assert(sizeof(float32_t) == 4);
    static_assert(sizeof(float64_t) == 8);
    static_assert(std::is_floating_point_v<float16_t>);
    static_assert(std::is_floating_point_v<float32_t>);
    static_assert(std::is_floating_point_v<float64_t>);


    /// Runtime identifier of arrow data types, usually associated with raw bytes with the associated value.
    // TODO: does not support all types specified by the Arrow specification
    // yet
    enum class data_type
    {
        NA = 0,
        BOOL,
        UINT8,
        INT8,
        UINT16,
        INT16,
        UINT32,
        INT32,
        UINT64,
        INT64,
        HALF_FLOAT,
        FLOAT,
        DOUBLE,
        // UTF8 variable-length string
        STRING,
        // Variable-length bytes (no guarantee of UTF8-ness)
        BINARY,
        // Fixed-size binary. Each value occupies the same number of bytes
        FIXED_SIZE_BINARY,
    };

    /// C++ types value representation types matching Arrow types.
    // NOTE: this needs to be in sync-order with `data_type`
    using all_base_types_t = mpl::typelist
    <
        std::nullopt_t  // REVIEW: not sure about if we need to have this one? for representing NA? is this the right type?
    ,   bool
    ,   std::uint8_t
    ,   std::int8_t
    ,   std::uint16_t
    ,   std::int16_t
    ,   std::uint32_t
    ,   std::int32_t
    ,   std::uint64_t
    ,   std::int64_t
    ,   float16_t
    ,   float32_t
    ,   float64_t
    ,   std::string
    //,   std::vector<std::byte> // REVIEW should this be uint8_t? char? buffer<unit8_t>?
    // TODO: add missing fundamental types here
    >;

    /// Type list of every C++ representation types supported by default, in order matching `data_type` related values.
    static constexpr all_base_types_t all_base_types;

    /// Matches C++ representation types which are supported by default.
    template< class T >
    concept is_arrow_base_type = mpl::contains<T>(all_base_types);

    /// Provides compile-time information about Arrow data types.
    /// Custom types can be made compatible by implementing this traits type.
    ///
    /// The follwing information must be provided if the type is an arrow type:
    /// - type_id : the runtime identifier value for that type, see `data_type`
    /// - value_type : the value representation type to use in C++ (usually T)
    /// - default_layout: the layout to use for that type FIXME: be more precise than that
    /// - MORE TO COME SOON, SEE TODOs BELOW
    ///
    /// @tparam Type for C++ value-representation that this type describes the traits of.
    ///
    /// @note: See ./arrow_traits.hpp for implementations for default base types.
/// /// @see `is_arrow_traits`, `has_arrow_type_traits`
    template<class T>
    struct arrow_traits;

    /// Matches valid and complete `arrow_traits` specializations for type T.
    /// Every type that needs to be compatible with this library's interface must
    /// provide a specialization of `arrow_traits`
    /// @see `arrow_traits`, `has_arrow_type_traits`
    template<class T>
    concept is_arrow_traits = mpl::is_type_instance_of_v< T, arrow_traits >
        and requires
        {
            /// Must proivde a compile-time value of type `data_type`.
            /// This is used to identify which arrow data type is represented in `value_type`
            requires std::same_as< std::remove_cvref_t<decltype(T::type_id)>, ::sparrow::data_type >;

            /// The C++ representation of the arrow value. For `arrow_traits<X>`, this is usually `X`.
            typename T::value_type;

            /// The arrow (binary) layout to use by default for representing a set of data for that type.
            typename T::default_layout;

            // TODO: add more interface requirements on the traits here
            // TODO: add conversion operations between bytes and the value type
        }
        ;


    /// Matches types providing valid and complete `arrow_traits` specialization.
    /// @see `is_arrow_traits`, `arrow_traits`
    template< class T >
    concept has_arrow_type_traits =
        requires { typename ::sparrow::arrow_traits<T>; }
        and is_arrow_traits< ::sparrow::arrow_traits<T> >
        ;

    /// Matches any type which is one of the base C++ types supported or at least that provides an `arrow_traits` specialization.
    template< class T >
    concept any_arrow_type = is_arrow_base_type<T> or has_arrow_type_traits<T>;

    /// @returns Arrow type id to use for a given C++ representation of that type.
    /// @see `arrow_traits`
    template< has_arrow_type_traits T >
    constexpr
    auto arrow_type_id() -> data_type
    {
        return arrow_traits<T>::type_id;
    }

    /// @returns Arrow type id to use for the type of a given object.
    /// @see `arrow_traits`
    template< has_arrow_type_traits T >
    constexpr
    auto arrow_type_id(const T&) -> data_type
    {
        return arrow_type_id<T>();
    }

    // TODO:
    // template< data_type id >
    // using default_type = mpl::at_t< mpl::find_if(all_base_types_t{}, mpl::same_as{})>;

    /// Binary layout type to use by default for the given C++ representation T of an arrow value.
    template< has_arrow_type_traits T >
    using default_layout_t = typename arrow_traits<T>::default_layout;


    // For now, a tiny wrapper around data_type
    // More data and functions to come
    class data_descriptor
    {
    public:

        constexpr data_descriptor()
            : data_descriptor(data_type::UINT8)
        {
        }

        constexpr explicit data_descriptor(data_type id)
            : m_id(id)
        {
        }

        constexpr data_type id() const { return m_id; }

    private:

        data_type m_id;
    };



}

