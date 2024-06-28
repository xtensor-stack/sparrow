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

#include <cstdint>
#include <cstdio>
#include <vector>

#include "sparrow/array_data_factory.hpp"
#include "sparrow/variable_size_binary_layout.hpp"

#include "doctest/doctest.h"

namespace sparrow
{
    TEST_SUITE("array_data_factory")
    {
        TEST_CASE("fixed_size_layout")
        {
            const std::vector<int32_t> v = {1, 2, 3, 4, 5};
            const dynamic_bitset<std::uint8_t> bitmap(v.size(), true);
            array_data ar = make_default_array_data<fixed_size_layout<int32_t>>(v, bitmap, 1);
            CHECK_EQ(ar.type.id(), data_descriptor(arrow_type_id<int32_t>()).id());
            CHECK_EQ(ar.length, 5);
            CHECK_EQ(ar.offset, 1);
            CHECK_EQ(ar.bitmap.size(), 5);
            CHECK_EQ(ar.buffers.size(), 1);
            CHECK_EQ(ar.buffers[0].size(), v.size() * sizeof(int32_t) / sizeof(uint8_t));
            auto data = ar.buffers[0].data<int32_t>();
            for (size_t i = 0; i < v.size(); ++i)
            {
                CHECK_EQ(data[i], v[i]);
            }
            CHECK_EQ(ar.child_data.size(), 0);
            CHECK_FALSE(ar.dictionary.has_value());

            fixed_size_layout<int32_t> layout(ar);
            CHECK_EQ(layout.size(), v.size() - 1);
        }

        TEST_CASE("variable_size_binary_layout")
        {
            constexpr size_t offset = 1;
            using Layout = variable_size_binary_layout<std::string, const std::string_view>;
            const std::vector<std::string> v = {"a", "bb", "ccc", "dddd", "eeeee"};
            const dynamic_bitset<std::uint8_t> bitmap(v.size(), true);
            array_data ar = make_default_array_data<Layout>(v, bitmap, offset);
            CHECK_EQ(ar.type.id(), data_descriptor(arrow_type_id<std::string>()).id());
            CHECK_EQ(ar.length, v.size());
            CHECK_EQ(ar.offset, offset);
            CHECK_EQ(ar.bitmap.size(), 5);
            CHECK_EQ(ar.buffers.size(), 2);

            CHECK_EQ(ar.child_data.size(), 0);
            CHECK_FALSE(ar.dictionary.has_value());

            Layout layout(ar);
            CHECK_EQ(layout.size(), v.size() - offset);
            for (size_t i = 0; i < layout.size(); ++i)
            {
                CHECK_EQ(layout[i].value(), v[i + offset]);
            }
        }

        TEST_CASE("dictionary_encoded_layout")
        {
            constexpr size_t offset = 1;
            using SubLayout = variable_size_binary_layout<std::string, const std::string_view>;
            using Layout = dictionary_encoded_layout<size_t, SubLayout>;
            const std::vector<std::string> v = {"a", "bb", "ccc", "bb", "a"};
            const dynamic_bitset<std::uint8_t> bitmap(v.size(), true);
            array_data ar = make_default_array_data<Layout>(v, bitmap, offset);
            CHECK_EQ(ar.type.id(), data_descriptor(arrow_type_id<std::uint64_t>()).id());
            CHECK_EQ(ar.length, 5);
            CHECK_EQ(ar.offset, offset);
            CHECK_EQ(ar.bitmap.size(), bitmap.size());
            CHECK_EQ(ar.buffers.size(), 1);
            CHECK_EQ(ar.child_data.size(), 0);
            REQUIRE(ar.dictionary.has_value());
            CHECK_EQ(ar.dictionary->type.id(), data_descriptor(arrow_type_id<std::string>()).id());
            CHECK_EQ(ar.dictionary->length, 3);
            CHECK_EQ(ar.dictionary->offset, 0);
            CHECK_EQ(ar.dictionary->bitmap.size(), 3);
            REQUIRE_EQ(ar.dictionary->buffers.size(), 2);
            CHECK_EQ(ar.buffers[0].size(), v.size() * sizeof(size_t) / sizeof(uint8_t));
            CHECK(ar.dictionary->child_data.empty());
            CHECK_FALSE(ar.dictionary->dictionary.has_value());

            SubLayout sublayout(*ar.dictionary);
            Layout layout(ar);
            CHECK_EQ(layout.size(), v.size() - offset);
            for (size_t i = 0; i < layout.size(); ++i)
            {
                CHECK_EQ(layout[i].value(), v[i + offset]);
            }
        }
    }
}
