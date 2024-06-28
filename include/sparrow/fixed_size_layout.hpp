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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or mplied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <utility>

#include "sparrow/array_data.hpp"
#include "sparrow/buffer.hpp"
#include "sparrow/contracts.hpp"
#include "sparrow/dynamic_bitset.hpp"
#include "sparrow/iterator.hpp"

namespace sparrow
{
    /**
     * A contiguous layout for fixed size types.
     *
     * This class provides a contiguous layout for fixed size types, such as `uint8_t`, `int32_t`, etc.
     * It iterates over the first buffer in the array_data, and uses the bitmap to skip over null.
     * The bitmap is assumed to be present in the array_data.
     *
     * @tparam T The type of the elements in the layout's data buffer.
     *           A fixed size type, such as a primitive type.
     *
     */
    template <class T>
    class fixed_size_layout
    {
    public:

        using self_type = fixed_size_layout<T>;
        using inner_value_type = T;
        using inner_reference = inner_value_type&;
        using inner_const_reference = const inner_value_type&;
        using const_inner_reference = const inner_value_type&;
        using bitmap_type = array_data::bitmap_type;
        using bitmap_reference = typename bitmap_type::reference;
        using bitmap_const_reference = typename bitmap_type::const_reference;
        using value_type = std::optional<inner_value_type>;
        using reference = reference_proxy<self_type>;
        using const_reference = const_reference_proxy<self_type>;
        using pointer = inner_value_type*;
        using const_pointer = const inner_value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_tag = std::contiguous_iterator_tag;

        using const_bitmap_iterator = array_data::bitmap_type::const_iterator;
        using const_value_iterator = pointer_iterator<const_pointer>;

        using bitmap_iterator = array_data::bitmap_type::iterator;
        using value_iterator = pointer_iterator<pointer>;

        using const_bitmap_range = std::ranges::subrange<const_bitmap_iterator>;
        using const_value_range = std::ranges::subrange<const_value_iterator>;

        using iterator = layout_iterator<self_type, false>;
        using const_iterator = layout_iterator<self_type, true>;

        explicit fixed_size_layout(array_data& data);
        void rebind_data(array_data& data);

        fixed_size_layout(const self_type&) = delete;
        self_type& operator=(const self_type&) = delete;
        fixed_size_layout(self_type&&) = delete;
        self_type& operator=(self_type&&) = delete;

        size_type size() const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        iterator begin();
        iterator end();

        const_iterator cbegin() const;
        const_iterator cend() const;

        const_bitmap_range bitmap() const;
        const_value_range values() const;

    private:

        pointer data();
        const_pointer data() const;

        bitmap_reference has_value(size_type i);
        bitmap_const_reference has_value(size_type i) const;
        inner_reference value(size_type i);
        inner_const_reference value(size_type i) const;

        value_iterator value_begin();
        value_iterator value_end();

        const_value_iterator value_cbegin() const;
        const_value_iterator value_cend() const;

        bitmap_iterator bitmap_begin();
        bitmap_iterator bitmap_end();

        const_bitmap_iterator bitmap_cbegin() const;
        const_bitmap_iterator bitmap_cend() const;

        array_data& data_ref();
        const array_data& data_ref() const;

        std::reference_wrapper<array_data> m_data;

        friend class reference_proxy<fixed_size_layout>;
        friend class const_reference_proxy<fixed_size_layout>;
    };

    /************************************
     * fixed_size_layout implementation *
     ***********************************/

    template <class T>
    fixed_size_layout<T>::fixed_size_layout(array_data& data)
        : m_data(data)
    {
        // We only require the presence of the bitmap and the first buffer.
        SPARROW_ASSERT_TRUE(data_ref().buffers.size() > 0);
        SPARROW_ASSERT_TRUE(static_cast<size_type>(data_ref().length) == data_ref().bitmap.size())
    }

    template <class T>
    void fixed_size_layout<T>::rebind_data(array_data& data)
    {
        m_data = data;
    }

    template <class T>
    auto fixed_size_layout<T>::size() const -> size_type
    {
        SPARROW_ASSERT_TRUE(data_ref().offset <= data_ref().length);
        return static_cast<size_type>(data_ref().length - data_ref().offset);
    }

    template <class T>
    auto fixed_size_layout<T>::value(size_type i) -> inner_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return data()[i + static_cast<size_type>(data_ref().offset)];
    }

    template <class T>
    auto fixed_size_layout<T>::value(size_type i) const -> inner_const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return data()[i + static_cast<size_type>(data_ref().offset)];
    }

    template <class T>
    auto fixed_size_layout<T>::operator[](size_type i) -> reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return reference(value(i), has_value(i));
    }

    template <class T>
    auto fixed_size_layout<T>::operator[](size_type i) const -> const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return const_reference(value(i), has_value(i));
    }

    template <class T>
    auto fixed_size_layout<T>::begin() -> iterator
    {
        return iterator(value_begin(), bitmap_begin());
    }

    template <class T>
    auto fixed_size_layout<T>::end() -> iterator
    {
        return iterator(value_end(), bitmap_end());
    }

    template <class T>
    auto fixed_size_layout<T>::cbegin() const -> const_iterator
    {
        return const_iterator(value_cbegin(), bitmap_cbegin());
    }

    template <class T>
    auto fixed_size_layout<T>::cend() const -> const_iterator
    {
        return const_iterator(value_cend(), bitmap_cend());
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap() const -> const_bitmap_range
    {
        return std::ranges::subrange(bitmap_cbegin(), bitmap_cend());
    }

    template <class T>
    auto fixed_size_layout<T>::values() const -> const_value_range
    {
        return std::ranges::subrange(value_cbegin(), value_cend());
    }

    template <class T>
    auto fixed_size_layout<T>::has_value(size_type i) -> bitmap_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return data_ref().bitmap[i + static_cast<size_type>(data_ref().offset)];
    }

    template <class T>
    auto fixed_size_layout<T>::has_value(size_type i) const -> bitmap_const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return data_ref().bitmap[i + static_cast<size_type>(data_ref().offset)];
    }

    template <class T>
    auto fixed_size_layout<T>::value_begin() -> value_iterator
    {
        return value_iterator{data() + data_ref().offset};
    }

    template <class T>
    auto fixed_size_layout<T>::value_end() -> value_iterator
    {
        value_iterator it = value_begin();
        std::advance(it, size());
        return it;
    }

    template <class T>
    auto fixed_size_layout<T>::value_cbegin() const -> const_value_iterator
    {
        return const_value_iterator{data() + data_ref().offset};
    }

    template <class T>
    auto fixed_size_layout<T>::value_cend() const -> const_value_iterator
    {
        auto it = value_cbegin();
        std::advance(it, size());
        return it;
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_begin() -> bitmap_iterator
    {
        return data_ref().bitmap.begin() + data_ref().offset;
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_end() -> bitmap_iterator
    {
        bitmap_iterator it = bitmap_begin();
        std::advance(it, size());
        return it;
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_cbegin() const -> const_bitmap_iterator
    {
        return data_ref().bitmap.cbegin() + data_ref().offset;
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_cend() const -> const_bitmap_iterator
    {
        const_bitmap_iterator it = bitmap_cbegin();
        std::advance(it, size());
        return it;
    }

    template <class T>
    auto fixed_size_layout<T>::data() -> pointer
    {
        SPARROW_ASSERT_TRUE(data_ref().buffers.size() > 0);
        return data_ref().buffers[0].template data<inner_value_type>();
    }

    template <class T>
    auto fixed_size_layout<T>::data() const -> const_pointer
    {
        SPARROW_ASSERT_TRUE(data_ref().buffers.size() > 0);
        return data_ref().buffers[0].template data<inner_value_type>();
    }

    template <class T>
    array_data& fixed_size_layout<T>::data_ref()
    {
        return m_data.get();
    }

    template <class T>
    const array_data& fixed_size_layout<T>::data_ref() const
    {
        return m_data.get();
    }

}  // namespace sparrow
