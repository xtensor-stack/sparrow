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
#include <iostream>
#include <numeric>

#include "doctest/doctest.h"

#include "sparrow/fixed_size_layout.hpp"

namespace sparrow
{
    // TODO: Test all the other base types once #15 is addressed.
    // static_assert(std::ranges::contiguous_range<fixed_size_layout<uint8_t>>);
    // static_assert(std::contiguous_iterator<fixed_size_layout_iterator<uint8_t, true>>);
    // static_assert(std::contiguous_iterator<fixed_size_layout_iterator<uint8_t, false>>);

    using data_type_t = int32_t;
    using layout_test_type = fixed_size_layout<data_type_t>;

    namespace
    {
        array_data make_test_array_data()
        {
            size_t n = 10;
            array_data ad;
            ad.type = data_descriptor(data_type::INT32);
            ad.bitmap = dynamic_bitset<uint8_t>(n, true);
            size_t buffer_size = (n * sizeof(data_type_t)) / sizeof(uint8_t);
            buffer<uint8_t> b(buffer_size);
            std::iota(b.data<int32_t>(), b.data<int32_t>() + n, -8);
            ad.buffers.push_back(b);
            ad.length = n;
            ad.offset = 1;
            ad.child_data.push_back(array_data());
            return ad;
        }

    }

    TEST_SUITE("fixed_size_layout")
    {
        TEST_CASE("constructors")
        {
            array_data ad = make_test_array_data();
            layout_test_type lt(ad);
            CHECK_EQ(lt.size(), ad.length - ad.offset);

            auto buffer_data = ad.buffers[0].data<data_type_t>();
            for (std::size_t i = 0; i < lt.size(); ++i)
            {
                CHECK_EQ(lt[i].value(), buffer_data[i + ad.offset]);
            }
        }

        TEST_CASE("value_iterator_ordering")
        {
            layout_test_type lt(make_test_array_data());
            auto lt_values = lt.values();
            layout_test_type::value_iterator iter = lt_values.begin();
            // TODO: Allow coercion of iterator to const_iterator.
            // layout_test_type::const_value_iterator citer = lt_values.begin();
            REQUIRE(iter < lt_values.end());
            // REQUIRE(citer < lt_values.end());
        }

        TEST_CASE("value_iterator_equality")
        {
            layout_test_type lt(make_test_array_data());
            auto lt_values = lt.values();
            layout_test_type::value_iterator iter = lt_values.begin();
            // TODO: Allow coercion of iterator to const_iterator.
            // layout_test_type::const_value_iterator citer = lt_values.begin();
            for (std::size_t i = 0; i < lt.size(); ++i)
            {
                CHECK_EQ(*iter++, lt[i]);
                // CHECK_EQ(*citer++, lt[i]);
            }
            CHECK_EQ(iter, lt_values.end());
            // CHECK_EQ(citer, lt_values.end());
        }

        TEST_CASE("iterator")
        {
            layout_test_type lt(make_test_array_data());
            auto it = lt.begin();
            auto end = lt.end();

            for (std::size_t i = 0; it != end; ++it, ++i)
            {
                CHECK_EQ(*it, std::make_optional(lt[i].value()));
                REQUIRE(it->has_value());
            }

            CHECK_EQ(it, end);
        }

    }

}
