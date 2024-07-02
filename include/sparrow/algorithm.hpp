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

#include <version> // required for standard library implementation pre-processor defines to be available
#include <algorithm>
#include <compare>
#include <concepts>

#include "sparrow/config.hpp"

namespace sparrow
{
#if COMPILING_WITH_APPLE_CLANG || USING_LIBCPP_PRE_17

    template <typename T>
    concept OrdCategory = std::same_as<T, std::strong_ordering> || std::same_as<T, std::weak_ordering>
                          || std::same_as<T, std::partial_ordering>;

    template <class R1, class R2, class Cmp>
    concept LexicographicalComparable = requires(const R1& r1, const R2& r2, Cmp comp) {
        { r1.cbegin() } -> std::input_or_output_iterator;
        { r2.cbegin() } -> std::input_or_output_iterator;
        OrdCategory<decltype(comp(*r1.cbegin(), *r2.cbegin()))>;
    };

    template <class R1, class R2, class Cmp>
        requires LexicographicalComparable<R1, R2, Cmp>
    constexpr auto lexicographical_compare_three_way_non_std(const R1& range1, const R2& range2, Cmp comp)
        -> decltype(comp(*range1.cbegin(), *range1.cbegin()))
    {
        auto iter_1 = range1.cbegin();
        const auto end_1 = range1.cend();
        auto iter_2 = range2.cbegin();
        const auto end_2 = range2.cend();

        while (true)
        {
            if (iter_1 == end_1)
            {
                return iter_2 == end_2 ? std::strong_ordering::equal : std::strong_ordering::less;
            }

            if (iter_2 == end_2)
            {
                return std::strong_ordering::greater;
            }

            if (const auto result = comp(*iter_1, *iter_2); result != 0)
            {
                return result;
            }

            ++iter_1;
            ++iter_2;
        }
    }
#endif

    template <class R1, class R2, class Cmp>
    constexpr auto lexicographical_compare_three_way(const R1& range1, const R2& range2, Cmp comp)
        -> decltype(comp(*range1.cbegin(), *range2.cbegin()))
    {
#if COMPILING_WITH_APPLE_CLANG || USING_LIBCPP_PRE_17
        return lexicographical_compare_three_way_non_std(range1, range2, comp);
#else
        return std::lexicographical_compare_three_way(
            range1.cbegin(),
            range1.cend(),
            range2.cbegin(),
            range2.cend(),
            comp
        );
#endif
    }

#if COMPILING_WITH_APPLE_CLANG || USING_LIBCPP_PRE_17
    struct compare_three_way
    {
        template <class T, class U>
        constexpr auto operator()(const T& t, const U& u) const noexcept -> std::partial_ordering
        {
            if (t < u)
            {
                return std::partial_ordering::less;
            }
            if (u < t)
            {
                return std::partial_ordering::greater;
            }
            return std::partial_ordering::equivalent;
        }
    };
#endif

    template <class R1, class R2>
    constexpr auto lexicographical_compare_three_way(const R1& r1, const R2& r2) -> std::partial_ordering
    {
        return lexicographical_compare_three_way<R1, R2>(
            r1,
            r2,
#if COMPILING_WITH_APPLE_CLANG || USING_LIBCPP_PRE_17
            compare_three_way {}
#else
            std::compare_three_way{}
#endif
        );
    }

    template <class R1, class R2>
    constexpr auto lexicographical_compare(const R1& r1, const R2& r2) -> bool
    {
        return lexicographical_compare_three_way(r1, r2) == std::strong_ordering::less;
    }

    template <std::input_iterator InputIt1, std::input_iterator InputIt2>
    constexpr
    bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
    {
        // see https://github.com/man-group/sparrow/issues/108
#if defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION < 180000)
        if(std::distance(first1, last1) != std::distance(first2, last2))
        {
            return false;
        }

        for(; first1 != last1; ++first1, ++first2){
            if (!(*first1 == *first2)){
                return false;
            }
        }

        return true;
#else
        return std::equal(first1, last1, first2, last2);
#endif
    }

}  // namespace sparrow
