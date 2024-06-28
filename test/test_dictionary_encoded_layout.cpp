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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>  // For doctest
#include <numeric>
#include <string_view>

#include "sparrow/dictionary_encoded_layout.hpp"
#include "sparrow/variable_size_binary_layout.hpp"

#include "doctest/doctest.h"

using data_type_t = uint8_t;
constexpr size_t element_count = 10;
static const std::array<data_type_t, element_count> indexes{1, 0, 3, 0, 1, 2, 3, 2, 4, 2};

namespace sparrow
{
    struct dictionary_encoded_fixture
    {
        dictionary_encoded_fixture()
        {
            m_data.type = data_descriptor(data_type::UINT8);
            m_data.bitmap = dynamic_bitset<uint8_t>(element_count, true);
            m_data.bitmap.set(9, false);
            constexpr size_t buffer_size = (element_count * sizeof(data_type_t)) / sizeof(uint8_t);
            buffer<uint8_t> b(buffer_size);
            std::ranges::copy(indexes, b.data<data_type_t>());
            m_data.buffers.push_back(b);
            m_data.length = element_count;
            auto dictionary = make_dictionary(words);
            m_data.dictionary = sparrow::value_ptr<array_data>(std::move(dictionary));
        }

        static array_data make_dictionary(const std::array<std::string_view, 5>& lwords)
        {
            array_data dictionary;
            dictionary.bitmap.resize(lwords.size());
            dictionary.buffers.resize(2);
            dictionary.buffers[0].resize(sizeof(std::int64_t) * (lwords.size() + 1));
            dictionary.buffers[1].resize(std::accumulate(
                lwords.cbegin(),
                lwords.cend(),
                size_t(0),
                [](std::size_t res, const auto& s)
                {
                    return res + s.size();
                }
            ));
            dictionary.buffers[0].data<std::int64_t>()[0] = 0u;
            auto iter = dictionary.buffers[1].begin();
            const auto offset = [&dictionary]()
            {
                return dictionary.buffers[0].data<std::int64_t>();
            };

            for (size_t i = 0; i < lwords.size(); ++i)
            {
                offset()[i + 1] = offset()[i] + static_cast<std::int64_t>(lwords[i].size());
                std::ranges::copy(lwords[i], iter);
                iter += static_cast<array_data::buffer_type::difference_type>(lwords[i].size());
                dictionary.bitmap.set(i, true);
            }
            dictionary.bitmap.set(4, false);

            dictionary.length = static_cast<int64_t>(lwords.size());
            dictionary.offset = 0;
            return dictionary;
        }

        static constexpr std::array<std::string_view, 5> words{{"you", "are", "not", "prepared", "null"}};

        array_data m_data;
        using sub_layout_type = variable_size_binary_layout<std::string, std::string_view>;
        using layout_type = dictionary_encoded_layout<data_type_t, sub_layout_type>;
    };

    TEST_SUITE("dictionary_encoded_layout")
    {
        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "constructors")
        {
            CHECK(m_data.buffers.size() == 1);
            const layout_type l_copy(m_data);
            CHECK(m_data.buffers.size() == 1);
        }

        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "rebind_data")
        {
            array_data data2 = m_data;
            layout_type l(m_data);
            static constexpr std::array<std::string_view, 5> new_words = {
                {"Just", "got", "home", "from", "Illinois"}
            };
            data2.dictionary = sparrow::value_ptr<array_data>(make_dictionary(new_words));
            l.rebind_data(data2);
            CHECK_EQ(l[0].value(), new_words[1]);
            CHECK_EQ(l[1].value(), new_words[0]);
            CHECK_EQ(l[2].value(), new_words[3]);
            CHECK_EQ(l[3].value(), new_words[0]);
            CHECK_EQ(l[4].value(), new_words[1]);
            CHECK_EQ(l[5].value(), new_words[2]);
            CHECK_EQ(l[6].value(), new_words[3]);
            CHECK_EQ(l[7].value(), new_words[2]);
            CHECK_FALSE(l[8].has_value());
            CHECK_FALSE(l[9].has_value());
        }

        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "size")
        {
            const layout_type l(m_data);
            CHECK_EQ(l.size(), element_count);
        }

        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "operator[]")
        {
            const layout_type l(m_data);
            CHECK_EQ(l[0].value(), words[1]);
            CHECK_EQ(l[1].value(), words[0]);
            CHECK_EQ(l[2].value(), words[3]);
            CHECK_EQ(l[3].value(), words[0]);
            CHECK_EQ(l[4].value(), words[1]);
            CHECK_EQ(l[5].value(), words[2]);
            CHECK_EQ(l[6].value(), words[3]);
            CHECK_EQ(l[7].value(), words[2]);
            CHECK_FALSE(l[8].has_value());
            CHECK_FALSE(l[9].has_value());
        }

        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "const_iterator")
        {
            const layout_type l(m_data);
            auto iter = l.cbegin();
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[0]);
            ++iter;
            --iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[0]);
            iter += 2;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[2]);
            ++iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[3]);
            ++iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[4]);
            ++iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[5]);
            ++iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[6]);
            ++iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[7]);
            ++iter;
            CHECK(iter->has_value());
            CHECK_EQ(iter->value(), l[8]);
            ++iter;
            CHECK_FALSE(iter->has_value());
            ++iter;
            CHECK_EQ(iter, l.cend());
        }

        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "const_value_iterator")
        {
            layout_type l(m_data);
            const auto vrange = l.values();
            auto iter = vrange.begin();
            CHECK_EQ(iter->value(), l[0].value());
            ++iter;
            --iter;
            CHECK_EQ(iter->value(), l[0].value());
            iter += 2;
            CHECK_EQ(iter->value(), l[2].value());
            ++iter;
            CHECK_EQ(iter->value(), l[3].value());
            ++iter;
            CHECK_EQ(iter->value(), l[4].value());
            ++iter;
            CHECK_EQ(iter->value(), l[5].value());
            ++iter;
            CHECK_EQ(iter->value(), l[6].value());
            ++iter;
            CHECK_EQ(iter->value(), l[7].value());
            ++iter;
            CHECK_EQ(iter->has_value(), l[8].has_value());
            ++iter;
            CHECK_EQ(iter->value(), words[2]);
            ++iter;
            CHECK_EQ(iter, vrange.end());
        }

        TEST_CASE_FIXTURE(dictionary_encoded_fixture, "const_bitmap_iterator")
        {
            const layout_type l(m_data);
            const auto brange = l.bitmap();
            auto iter = brange.begin();
            CHECK(*iter);
            ++iter;
            CHECK(*iter);
            iter += 8;
            CHECK(!*iter);
            ++iter;
            CHECK_EQ(iter, brange.end());
        }
    }
}
