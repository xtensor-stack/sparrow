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

#include <array>
#include <iostream>
#include <numeric>
#include <string_view>

#include "sparrow/array_data.hpp"
#include "sparrow/array_data_factory.hpp"
#include "sparrow/contracts.hpp"
#include "sparrow/variable_size_binary_layout.hpp"

#include "doctest/doctest.h"

namespace sparrow
{
    struct vs_binary_fixture
    {
        using layout_type = variable_size_binary_layout<std::string, std::string_view>;

        vs_binary_fixture()
        {
            array_data::bitmap_type bitmap{words.size(), true};
            bitmap.set(2, false);
            m_data = make_default_array_data<layout_type>(words, bitmap, 1);
        }

        static constexpr std::array<std::string_view, 4> words = {"you", "are", "not", "prepared"};
        array_data m_data;

    private:

       // TODO This isn't used at the moment, are we supposed to use it later or should we remove it to avoid polluting the file?
//         std::int64_t* offset()
//         {
//             SPARROW_ASSERT_FALSE(m_data.buffers.empty());
//             return m_data.buffers[0].data<std::int64_t>();
//         }

        static_assert(std::same_as<layout_type::inner_value_type, std::string>);
        static_assert(std::same_as<layout_type::inner_reference, sparrow::vs_binary_reference<layout_type>>);
        static_assert(std::same_as<layout_type::inner_const_reference, std::string_view>);
        using const_value_iterator = layout_type::const_value_iterator;
        static_assert(std::same_as<const_value_iterator::value_type, std::string>);
        static_assert(std::same_as<const_value_iterator::reference, std::string_view>);
    };

    TEST_SUITE("variable_size_binary_layout")
    {
        TEST_CASE_FIXTURE(vs_binary_fixture, "rebind_data")
        {
            layout_type l(m_data);
            static constexpr std::array<std::string_view, 3> new_words = {"tambourines", "and", "elephant"};
            array_data::bitmap_type new_bitmap{new_words.size(), true};
            auto data = make_default_array_data<layout_type>(new_words, new_bitmap, 0);
            l.rebind_data(data);
            for (std::size_t i = 0; i < new_words.size(); ++i)
            {
                CHECK_EQ(l[i].value(), new_words[i]);
            }
        }

        TEST_CASE_FIXTURE(vs_binary_fixture, "size")
        {
            layout_type l(m_data);
            CHECK_EQ(l.size(), m_data.length - m_data.offset);
        }

        TEST_CASE_FIXTURE(vs_binary_fixture, "operator[]")
        {
            layout_type l(m_data);
            auto cref0 = l[0];
            auto cref1 = l[1];
            auto cref2 = l[2];

            CHECK_EQ(cref0.value(), words[1]);
            CHECK(!cref1.has_value());
            CHECK_EQ(cref2.value(), words[3]);

            l[0].value() = std::string("is");
            l[2].value() = std::string("unpreparedandmore");

            CHECK_EQ(cref0.value(), std::string("is"));
            CHECK(!cref1.has_value());
            CHECK_EQ(cref2.value(), std::string("unpreparedandmore"));

            l[0].value() = std::string("are");
            l[2].value() = std::string("ok");

            CHECK_EQ(cref0.value(), std::string("are"));
            CHECK(!cref1.has_value());
            CHECK_EQ(cref2.value(), std::string("ok"));
        }

        TEST_CASE_FIXTURE(vs_binary_fixture, "const_value_iterator")
        {
            layout_type l(m_data);
            auto cref0 = l[0];
            auto cref2 = l[2];

            auto vrange = l.values();
            auto iter = vrange.begin();
            CHECK_EQ(*iter, cref0.value());
            ++iter;
            --iter;
            CHECK_EQ(*iter, cref0.value());
            iter += 2;
            CHECK_EQ(*iter, cref2.value());
            ++iter;
            CHECK_EQ(iter, vrange.end());
        }

        TEST_CASE_FIXTURE(vs_binary_fixture, "const_bitmap_iterator")
        {
            layout_type l(m_data);
            auto brange = l.bitmap();
            auto iter = brange.begin();
            CHECK(*iter);
            ++iter;
            CHECK(!*iter);
            iter += 2;
            CHECK_EQ(iter, brange.end());
        }

        TEST_CASE_FIXTURE(vs_binary_fixture, "const_iterator")
        {
            const layout_type l(m_data);
            auto cref0 = l[0];
            auto cref2 = l[2];

            auto iter = l.cbegin();
            CHECK_EQ(*iter, std::make_optional(cref0.value()));

            ++iter;
            CHECK(!iter->has_value());

            ++iter;
            CHECK_EQ(iter->value(), cref2.value());

            iter++;
            CHECK_EQ(iter, l.cend());

            iter -= 3;
            CHECK_EQ(iter, l.cbegin());
        }

        TEST_CASE("vs_binary_reference")
        {
            using layout_type = variable_size_binary_layout<std::string, std::string_view>;

            constexpr std::array<std::string_view, 4> words = {"you", "are", "not", "prepared"};

            array_data::bitmap_type bitmap{words.size(), true};
            array_data vs_data = make_default_array_data<layout_type>(words, bitmap, 0);

            layout_type l(vs_data);
            CHECK_EQ(l.size(), vs_data.length - vs_data.offset);

            auto cref0 = l[0];
            auto cref1 = l[1];
            auto cref2 = l[2];
            auto cref3 = l[3];

            CHECK(cref0.has_value());
            CHECK_EQ(cref0.value(), words[0]);
            CHECK(cref1.has_value());
            CHECK_EQ(cref1.value(), words[1]);
            CHECK(cref2.has_value());
            CHECK_EQ(cref2.value(), words[2]);
            CHECK(cref3.has_value());
            CHECK_EQ(cref3.value(), words[3]);

            SUBCASE("inner_reference")
            {
                cref3.value() = std::string_view("unpreparedandmore");

                CHECK_EQ(cref0.value(), words[0]);
                CHECK_EQ(cref1.value(), words[1]);
                CHECK_EQ(cref2.value(), words[2]);
                CHECK_EQ(cref3.value(), std::string("unpreparedandmore"));

                cref0.value() = std::string("he");
                cref1.value() = std::string("is");
                cref2.value() = std::string("");

                CHECK_EQ(cref0.value(), std::string("he"));
                CHECK_EQ(cref1.value(), std::string("is"));
                CHECK_EQ(cref2.value(), std::string_view(""));
                CHECK_EQ(cref3.value(), std::string("unpreparedandmore"));
            }

            SUBCASE("operator==self_type")
            {
                vs_binary_reference<layout_type> vs_ref0(&l, 0);
                CHECK_EQ(cref0.value(), vs_ref0);

                vs_binary_reference<layout_type> vs_ref3(&l, 3);
                CHECK_EQ(cref3.value(), vs_ref3);
            }

            SUBCASE("operator=self_type")
            {
                constexpr std::array<std::string_view, 4> replacement_words = {"this", "is", "a", "replacement"};

                array_data::bitmap_type rpl_bitmap{replacement_words.size(), true};
                array_data rpl_vs_data = make_default_array_data<layout_type>(replacement_words, rpl_bitmap, 0);

                layout_type rpl_l(rpl_vs_data);
                CHECK_EQ(rpl_l.size(), rpl_vs_data.length - rpl_vs_data.offset);

                vs_binary_reference<layout_type> rpl_vs_ref0(&rpl_l, 0);
                vs_binary_reference<layout_type> rpl_vs_ref1(&rpl_l, 1);
                vs_binary_reference<layout_type> rpl_vs_ref2(&rpl_l, 2);
                vs_binary_reference<layout_type> rpl_vs_ref3(&rpl_l, 3);

                cref0.value() = std::move(rpl_vs_ref0);
                cref1.value() = std::move(rpl_vs_ref1);
                cref2.value() = std::move(rpl_vs_ref2);
                cref3.value() = std::move(rpl_vs_ref3);

                CHECK_EQ(cref0.value(), std::string("this"));
                CHECK_EQ(cref1.value(), std::string("is"));
                CHECK_EQ(cref2.value(), std::string("a"));
                CHECK_EQ(cref3.value(), std::string("replacement"));
            }
        }
    }
}
