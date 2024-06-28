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

#include <initializer_list>

#include "sparrow/array_data.hpp"
#include "sparrow/contracts.hpp"

#include "doctest/doctest.h"

namespace sparrow
{
    class mock_layout
    {
    public:

        using value_type = std::optional<int>;
        using inner_value_type = value_type::value_type;
        using inner_reference = int&;
        using inner_const_reference = const int&;
        using const_inner_reference = const int&;
        using bitmap_type = array_data::bitmap_type;
        using bitmap_reference = typename bitmap_type::reference;
        using bitmap_const_reference = typename bitmap_type::const_reference;
        using reference = reference_proxy<mock_layout>;
        using const_reference = const_reference_proxy<mock_layout>;
        using size_type = std::size_t;

        mock_layout() = default;

        mock_layout(std::initializer_list<value_type> l)
            : m_bitmap(l.size())
            , m_data(l.size())
        {
            auto bit_iter = m_bitmap.begin();
            auto value_iter = m_data.begin();
            for (const auto& v : l)
            {
                *bit_iter++ = v.has_value();
                if (v.has_value())
                {
                    *value_iter++ = v.value();
                }
                else
                {
                    ++value_iter;
                }
            }
        }

        reference operator[](size_type pos)
        {
            SPARROW_ASSERT_TRUE(pos < m_bitmap.size());
            SPARROW_ASSERT_TRUE(pos < m_data.size());
            return reference(m_data[pos], m_bitmap[pos]);
        }

        const_reference operator[](size_type pos) const
        {
            SPARROW_ASSERT_TRUE(pos < m_bitmap.size());
            SPARROW_ASSERT_TRUE(pos < m_data.size());
            return const_reference(m_data[pos], m_bitmap[pos]);
        }

        bool has_value(size_type pos) const
        {
            SPARROW_ASSERT_TRUE(pos < m_bitmap.size());
            return m_bitmap[pos];
        }

        inner_const_reference value(size_type pos) const
        {
            SPARROW_ASSERT_TRUE(pos < m_data.size());
            return m_data[pos];
        }

    private:

        bitmap_type m_bitmap;
        std::vector<int> m_data;
    };

    struct ref_proxy_fixture
    {
        ref_proxy_fixture()
        {
            m_layout = {std::make_optional(2), std::make_optional(5), std::nullopt, std::make_optional(7)};
        }

        const mock_layout& layout() const
        {
            return m_layout;
        }

        mock_layout m_layout;
    };

    TEST_SUITE("reference_proxy")
    {
        TEST_CASE_FIXTURE(ref_proxy_fixture, "has_value")
        {
            auto ref0 = m_layout[0];
            CHECK(ref0.has_value());
            CHECK(ref0);

            auto ref2 = m_layout[2];
            CHECK(!ref2.has_value());
            CHECK(!ref2);
        }

        TEST_CASE_FIXTURE(ref_proxy_fixture, "value")
        {
            auto ref0 = m_layout[0];
            CHECK_EQ(ref0.value(), m_layout.value(0));
            static_assert(std::same_as<decltype(ref0.value()), int&>);

            const auto cref0 = m_layout[0];
            CHECK_EQ(cref0.value(), m_layout.value(0));
            static_assert(std::same_as<decltype(cref0.value()), const int&>);
        }

        TEST_CASE_FIXTURE(ref_proxy_fixture, "assign")
        {
            int expected_value0 = 4;
            auto ref0 = m_layout[0];
            ref0 = expected_value0;
            CHECK_EQ(m_layout.value(0), expected_value0);

            ref0 = std::nullopt;
            CHECK(!ref0.has_value());

            ref0 = expected_value0;
            CHECK_EQ(m_layout.value(0), expected_value0);

            int expected_value2 = 3;
            auto ref2 = m_layout[2];
            ref2 = std::make_optional(expected_value2);
            CHECK_EQ(m_layout.value(2), expected_value2);

            ref0 = ref2;
            CHECK_EQ(m_layout.value(0), expected_value2);

            int expected_value3 = 8;
            std::optional<int> opt(expected_value3);
            auto ref3 = m_layout[3];
            ref3 = opt;
            CHECK_EQ(m_layout.value(3), expected_value3);

            ref0 = std::move(ref3);
            CHECK_EQ(m_layout.value(0), expected_value3);
        }

        TEST_CASE_FIXTURE(ref_proxy_fixture, "swap")
        {
            int expected_value0 = m_layout.value(0);
            int expected_value3 = m_layout.value(3);

            auto ref0 = m_layout[0];
            auto ref2 = m_layout[2];
            auto ref3 = m_layout[3];

            ref0.swap(ref2);
            CHECK(!ref0);
            CHECK(ref2);
            CHECK_EQ(ref2.value(), expected_value0);

            swap(ref2, ref3);
            CHECK_EQ(ref2.value(), expected_value3);
            CHECK_EQ(ref3.value(), expected_value0);
        }

        TEST_CASE_FIXTURE(ref_proxy_fixture, "comparison")
        {
            SUBCASE("equality")
            {
                auto ref0 = m_layout[0];
                auto ref1 = m_layout[1];
                auto ref2 = m_layout[2];

                CHECK(ref0 != ref1);
                CHECK(ref0 != m_layout.value(1));
                CHECK(m_layout.value(0) != ref1);
                ref0 = ref1;
                CHECK(ref0 == ref1);
                CHECK(ref0 == m_layout.value(1));
                CHECK(m_layout.value(0) == ref1);

                CHECK(ref0 != ref2);
            }

            SUBCASE("inequality")
            {
                auto ref0 = m_layout[0];
                auto ref2 = m_layout[2];
                auto ref3 = m_layout[3];

                CHECK(ref2 < ref0);
                CHECK(ref2 <= ref0);
                CHECK(ref3 > ref2);
                CHECK(ref3 >= ref2);

                CHECK(ref0 < ref3);
                CHECK(ref0 <= ref3);
                CHECK(ref0.value() < ref3);
                CHECK(ref0.value() <= ref3);
                CHECK(ref0 < ref3.value());
                CHECK(ref0 <= ref3.value());

                CHECK(ref3 > ref0);
                CHECK(ref3 >= ref0);
                CHECK(ref3.value() > ref0);
                CHECK(ref3.value() >= ref0);
                CHECK(ref3 > ref0.value());
                CHECK(ref3 >= ref0.value());
            }
        }
    }

    TEST_SUITE("const_reference_proxy")
    {
        TEST_CASE_FIXTURE(ref_proxy_fixture, "has_value")
        {
            auto ref0 = layout()[0];
            CHECK(ref0.has_value());
            CHECK(ref0);

            auto ref2 = layout()[2];
            CHECK(!ref2.has_value());
            CHECK(!ref2);
        }

        TEST_CASE_FIXTURE(ref_proxy_fixture, "value")
        {
            auto ref0 = layout()[0];
            CHECK_EQ(ref0.value(), m_layout.value(0));
            static_assert(std::same_as<decltype(ref0.value()), const int&>);
        }
    }
}
