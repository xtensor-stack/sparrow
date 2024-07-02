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
#include <cstddef>
#include <numeric>

#include "sparrow/buffer.hpp"
#include "sparrow/iterator.hpp"

#include "doctest/doctest.h"

namespace sparrow
{

    class test_iterator : public iterator_base<test_iterator, int, std::contiguous_iterator_tag>
    {
    public:

        explicit test_iterator(int* p = nullptr)
            : m_pointer(p)
        {
        }

    private:

        using self_type = test_iterator;
        using base_type = iterator_base<self_type, int, std::contiguous_iterator_tag>;
        using reference = typename base_type::reference;
        using difference_type = typename base_type::difference_type;

        reference dereference() const
        {
            return *m_pointer;
        }

        void increment()
        {
            ++m_pointer;
        }

        void decrement()
        {
            --m_pointer;
        }

        void advance(difference_type n)
        {
            m_pointer += n;
        }

        difference_type distance_to(const self_type& rhs) const
        {
            return rhs.m_pointer - m_pointer;
        }

        bool equal(const self_type& rhs) const
        {
            return m_pointer == rhs.m_pointer;
        }

        bool less_than(const self_type& rhs) const
        {
            return m_pointer < rhs.m_pointer;
        }

        int* m_pointer;

        friend class iterator_access;
    };

    namespace
    {
        buffer<int> make_test_buffer()
        {
            constexpr std::size_t size = 16u;
            buffer<int> res(size);
            std::iota(res.data(), res.data() + size, 0);
            return res;
        }
    }

    struct iterator_fixture
    {
        iterator_fixture()
            : buff(make_test_buffer())
            , iter(buff.data())
        {
        }

        buffer<int> buff;
        test_iterator iter;
    };

    TEST_SUITE("iterator_base")
    {
        TEST_CASE("value semantic")
        {
            auto buff = make_test_buffer();

            SUBCASE("constructor")
            {
                {
                    [[maybe_unused]] test_iterator it;
                }

                {
                    test_iterator it(buff.data());
                    CHECK_EQ(it.operator->(), buff.data());
                }
            }

            SUBCASE("copy semantic")
            {
                test_iterator it1(buff.data());
                test_iterator it2(it1);
                CHECK_EQ(it1, it2);

                auto buff2 = make_test_buffer();
                test_iterator it3(buff2.data());
                CHECK_NE(it1, it3);
                it3 = it1;
                CHECK_EQ(it1, it3);
            }

            SUBCASE("move semantic")
            {
                test_iterator it1(buff.data());
                test_iterator it2(std::move(it1));
                CHECK_EQ(it2.operator->(), buff.data());

                auto buff2 = make_test_buffer();
                test_iterator it3(buff2.data());
                CHECK_NE(it2, it3);
                it3 = std::move(it2);
                CHECK_EQ(it3.operator->(), buff.data());
            }
        }

        TEST_CASE("iterator types")
        {
            constexpr bool value_type = std::same_as<std::iter_value_t<test_iterator>, int>;
            CHECK(value_type);

            constexpr bool reference_type = std::same_as<std::iter_reference_t<test_iterator>, int&>;
            CHECK(reference_type);

            constexpr bool difference_type = std::same_as<std::iter_difference_t<test_iterator>, std::ptrdiff_t>;
            CHECK(difference_type);

            constexpr bool rvalue_reference_type = std::same_as<std::iter_rvalue_reference_t<test_iterator>, int&&>;
            CHECK(rvalue_reference_type);

            constexpr bool common_reference_type = std::same_as<std::iter_common_reference_t<test_iterator>, int&>;
            CHECK(common_reference_type);
        }

        TEST_CASE_FIXTURE(iterator_fixture, "forward iterator")
        {
            constexpr bool valid = std::forward_iterator<test_iterator>;
            CHECK(valid);

            CHECK_EQ(*iter, buff.data()[0]);
            ++iter;
            CHECK_EQ(*iter, buff.data()[1]);

            auto iter2 = iter++;
            CHECK_EQ(*iter2, buff.data()[1]);
            CHECK_EQ(*iter, buff.data()[2]);

            CHECK_EQ(iter.operator->(), buff.data() + 2);

            auto iter3 = test_iterator(buff.data());
            ++iter3;
            ++iter3;
            CHECK_EQ(iter, iter3);
        }

        TEST_CASE_FIXTURE(iterator_fixture, "bidirectional iterator")
        {
            constexpr bool valid = std::bidirectional_iterator<test_iterator>;
            CHECK(valid);

            for (int i = 0; i < 5; ++i)
            {
                ++iter;
            }

            --iter;
            CHECK_EQ(*iter, buff.data()[4]);
            auto iter2 = iter--;
            CHECK_EQ(*iter2, buff.data()[4]);
            CHECK_EQ(*iter, buff.data()[3]);
        }

        TEST_CASE_FIXTURE(iterator_fixture, "random access iterator")
        {
            constexpr bool valid = std::random_access_iterator<test_iterator>;
            CHECK(valid);

            iter += 4;
            CHECK_EQ(*iter, buff.data()[4]);
            auto iter2 = iter + 2;
            CHECK_EQ(*iter, buff.data()[4]);
            CHECK_EQ(*iter2, buff.data()[6]);
            auto iter4 = 2 + iter2;
            CHECK_EQ(*iter4, buff.data()[8]);

            iter2 -= 2;
            CHECK_EQ(*iter2, buff.data()[4]);
            auto iter3 = iter2 - 3;
            CHECK_EQ(*iter2, buff.data()[4]);
            CHECK_EQ(*iter3, buff.data()[1]);

            auto diff = iter2 - iter3;
            CHECK_EQ(diff, 3);

            int r = iter3[5];
            CHECK_EQ(r, buff.data()[6]);

            CHECK(iter3 < iter2);
            CHECK(iter3 <= iter2);
            CHECK(iter2 > iter3);
            CHECK(iter2 >= iter3);
        }

        TEST_CASE_FIXTURE(iterator_fixture, "contiguous iterator")
        {
            #ifndef EMSCRIPTEN
            constexpr bool valid = std::contiguous_iterator<test_iterator>;
            CHECK(valid);
            #endif
        }
    }

    TEST_SUITE("pointer_iterator")
    {
        TEST_CASE("make_pointer_iterator")
        {
            std::array<int, 3> a = {2, 4, 6};
            auto iter = make_pointer_iterator(&a[0]);
            CHECK_EQ(*iter, a[0]);
            ++iter;
            CHECK_EQ(*iter, a[1]);
            ++iter;
            CHECK_EQ(*iter, a[2]);
        }

        TEST_CASE("const conversion")
        {
            std::array<int, 3> a = {2, 4, 6};
            using iterator = pointer_iterator<int*>;
            using const_iterator = pointer_iterator<const int*>;

            const_iterator iter{&(a[0])};
            CHECK_EQ(*iter, a[0]);

            iterator iter2{&(a[0])};
            CHECK_EQ(*iter2, a[0]);
            *iter2 = 3;
            CHECK_EQ(a[0], 3);
        }
    }
}
