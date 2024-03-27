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
#include <cassert>

#include "sparrow/array_data.hpp"
#include "sparrow/iterator.hpp"
#include "sparrow/buffer.hpp"
#include "sparrow/dynamic_bitset.hpp"

namespace sparrow
{
    /**
     * An iterator for `fixed_size_layout` operating on contiguous data only.
     *
     * @tparam T The type of the elements in the layout's data buffer.
     * @tparam is_const A boolean indicating whether the iterator is const.
     *
     */
    template <class T, bool is_const>
    class fixed_size_layout_value_iterator
        : public iterator_base
        <
            fixed_size_layout_value_iterator<T, is_const>,
            mpl::constify_t<T, is_const>,
            std::contiguous_iterator_tag
        >
    {
    public:

        using self_type = fixed_size_layout_value_iterator<T, is_const>;
        using base_type = iterator_base
        <
            self_type,
            mpl::constify_t<T, is_const>,
            std::contiguous_iterator_tag
        >;
        using pointer = typename base_type::pointer;

        fixed_size_layout_value_iterator() = default;
        explicit fixed_size_layout_value_iterator(pointer p);

    private:
        using reference = typename base_type::reference;
        using difference_type = typename base_type::difference_type;
        using size_type = std::size_t;

        reference dereference() const;
        void increment();
        void decrement();
        void advance(difference_type n);
        difference_type distance_to(const self_type& hs) const;
        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

        pointer m_pointer = nullptr;

        friend class iterator_access;
    };

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
        using inner_const_reference = const inner_reference;
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

        using const_bitmap_iterator = array_data::bitmap_type::const_iterator;
        using const_value_iterator = fixed_size_layout_value_iterator<T, true>;

        using bitmap_iterator = array_data::bitmap_type::iterator;
        using value_iterator = fixed_size_layout_value_iterator<T, false>;

        using const_bitmap_range = std::ranges::subrange<const_bitmap_iterator>;
        using const_value_range = std::ranges::subrange<const_value_iterator>;

        using bitmap_range = std::ranges::subrange<bitmap_iterator>;
        using value_range = std::ranges::subrange<value_iterator>;

        using iterator = layout_iterator<self_type, false>;
        using const_iterator = layout_iterator<self_type, true>;

        explicit fixed_size_layout(array_data p);

        size_type size() const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        iterator begin();
        iterator end();

        const_iterator cbegin() const;
        const_iterator cend() const;

        bitmap_range bitmap();
        value_range values();

        const_bitmap_range bitmap() const;
        const_value_range values() const;

    private:
        array_data m_data;

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

        friend class reference_proxy<fixed_size_layout>;
        friend class const_reference_proxy<fixed_size_layout>;
    };

    /***************************************************
     * fixed_size_layout_value_iterator implementation *
     **************************************************/

    template <class T, bool is_const>
    fixed_size_layout_value_iterator<T, is_const>::fixed_size_layout_value_iterator(pointer pointer)
        : m_pointer(pointer)
    {
    }

    template <class T, bool is_const>
    auto fixed_size_layout_value_iterator<T, is_const>::dereference() const -> reference
    {
        return *m_pointer;
    }

    template <class T, bool is_const>
    void fixed_size_layout_value_iterator<T, is_const>::increment()
    {
        ++m_pointer;
    }

    template <class T, bool is_const>
    void fixed_size_layout_value_iterator<T, is_const>::decrement()
    {
        --m_pointer;
    }

    template <class T, bool is_const>
    void fixed_size_layout_value_iterator<T, is_const>::advance(difference_type n)
    {
        m_pointer += n;
    }

    template <class T, bool is_const>
    auto fixed_size_layout_value_iterator<T, is_const>::distance_to(const self_type& rhs) const -> difference_type
    {
        return rhs.m_pointer - m_pointer;
    }

    template <class T, bool is_const>
    bool fixed_size_layout_value_iterator<T, is_const>::equal(const self_type& rhs) const
    {
        return rhs.m_pointer == m_pointer;
    }

    template <class T, bool is_const>
    bool fixed_size_layout_value_iterator<T, is_const>::less_than(const self_type& rhs) const
    {
        return m_pointer < rhs.m_pointer;
    }

    /************************************
     * fixed_size_layout implementation *
     ***********************************/

    template <class T>
    fixed_size_layout<T>::fixed_size_layout(array_data ad)
        : m_data(std::move(ad))
    {
        // We only require the presence of the bitmap and the first buffer.
        assert(m_data.buffers.size() > 0);
        assert(m_data.length == m_data.bitmap.size());
    }

    template <class T>
    auto fixed_size_layout<T>::size() const -> size_type
    {
        assert(m_data.offset <= m_data.length);
        return static_cast<size_type>(m_data.length - m_data.offset);
    }

    template <class T>
    auto fixed_size_layout<T>::value(size_type i) -> inner_reference
    {
        assert(i < size());
        return data()[i + m_data.offset];
    }

    template <class T>
    auto fixed_size_layout<T>::value(size_type i) const -> inner_const_reference
    {
        assert(i < size());
        return data()[i + m_data.offset];
    }

    template <class T>
    auto fixed_size_layout<T>::operator[](size_type i) -> reference
    {
        assert(i < size());
        return reference(value(i), has_value(i));
    }

    template <class T>
    auto fixed_size_layout<T>::operator[](size_type i) const -> const_reference
    {
        assert(i < size());
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
    auto fixed_size_layout<T>::bitmap() -> bitmap_range
    {
        return std::ranges::subrange(bitmap_begin(), bitmap_end());
    }

    template <class T>
    auto fixed_size_layout<T>::values() -> value_range
    {
        return std::ranges::subrange(value_begin(), value_end());
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
        assert(i < size());
        return m_data.bitmap[i + m_data.offset];
    }

    template <class T>
    auto fixed_size_layout<T>::has_value(size_type i) const -> bitmap_const_reference
    {
        assert(i < size());
        return m_data.bitmap[i + m_data.offset];
    }

    template <class T>
    auto fixed_size_layout<T>::value_begin() -> value_iterator
    {
        return value_iterator{data() + m_data.offset};
    }

    template <class T>
    auto fixed_size_layout<T>::value_end() -> value_iterator
    {
        return value_begin() + size();
    }

    template <class T>
    auto fixed_size_layout<T>::value_cbegin() const -> const_value_iterator
    {
        return const_value_iterator{data() + m_data.offset};
    }

    template <class T>
    auto fixed_size_layout<T>::value_cend() const -> const_value_iterator
    {
        return value_cbegin() + size();
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_begin() -> bitmap_iterator
    {
        return m_data.bitmap.begin() + m_data.offset;
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_end() -> bitmap_iterator
    {
        return bitmap_begin() + size();
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_cbegin() const -> const_bitmap_iterator
    {
        return m_data.bitmap.cbegin() + m_data.offset;
    }

    template <class T>
    auto fixed_size_layout<T>::bitmap_cend() const -> const_bitmap_iterator
    {
        return bitmap_cbegin() + size();
    }

    template <class T>
    auto fixed_size_layout<T>::data() -> pointer
    {
        assert(m_data.buffers.size() > 0);
        return m_data.buffers[0].template data<inner_value_type>();
    }

    template <class T>
    auto fixed_size_layout<T>::data() const -> const_pointer
    {
        assert(m_data.buffers.size() > 0);
        return m_data.buffers[0].template data<inner_value_type>();
    }

} // namespace sparrow


