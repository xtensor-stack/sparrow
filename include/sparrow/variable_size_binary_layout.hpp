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

#include <ranges>
#include "sparrow/array_data.hpp"
#include "sparrow/iterator.hpp"

namespace sparrow
{
    namespace impl
    {
        template <class C, bool is_const>
        struct get_inner_reference
            : std::conditional<
                is_const,
                typename C::inner_const_reference,
                typename C::inner_reference
              >
        {
        };

        template <class C, bool is_const>
        using get_inner_reference_t = typename get_inner_reference<C, is_const>::type;
    }

    template <class T>
    concept layout_offset = std::same_as<T, std::int32_t> || std::same_as<T, std::int64_t>;

    /**
     * @class vs_binary_value_iterator
     *
     * @brief Iterator over the data values of a variable size binary
     * layout.
     *
     * @tparam L the layout type
     * @tparam is_const a boolean flag specifying whether this iterator is const.
     */
    template <class L, bool is_const>
    class vs_binary_value_iterator : public iterator_base
    <
        vs_binary_value_iterator<L, is_const>,
        mpl::constify_t<typename L::inner_value_type, is_const>,
        std::contiguous_iterator_tag,
        impl::get_inner_reference_t<L, is_const>
    >
    {
    public:

        using self_type = vs_binary_value_iterator<L, is_const>;
        using base_type = iterator_base
        <
            self_type,
            mpl::constify_t<typename L::inner_value_type, is_const>,
            std::contiguous_iterator_tag,
            impl::get_inner_reference_t<L, is_const>
        >;
        using reference = typename base_type::reference;
        using difference_type = typename base_type::difference_type;
        
        using offset_iterator = std::conditional_t<
            is_const, typename L::const_offset_iterator, typename L::offset_iterator
        >;
        using data_iterator = std::conditional_t<
            is_const, typename L::const_data_iterator, typename L::data_iterator
        >;

        vs_binary_value_iterator() noexcept = default;
        vs_binary_value_iterator(
            offset_iterator offset_it,
            data_iterator data_begin
        );

    private:

        reference dereference() const;
        void increment();
        void decrement();
        void advance(difference_type n);
        difference_type distance_to(const self_type& rhs) const;
        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

        offset_iterator m_offset_it;
        data_iterator m_data_begin;

        friend class iterator_access;
    };

    /*
     * @class variable_size_binary_layout
     *
     * @brief Layout for arrays containing values consisting of a variable number of bytes.
     *
     * This layout is used to retrieve data in an array of values of a variable number of bytes
     * (typically string objects). Values are stored contiguously in a data buffer (for instance
     * a buffer of char if values are strings), a single value is retrieved via an additional
     * offset buffer, where each element is the beginning of the corresponding value in the data
     * buffer.
     *
     * Example:
     *
     * Let's consider the array of string ['please', 'allow', 'me', 'to', 'introduce', 'myself'].
     * The internal buffers will be:
     * - offset: [0, 6, 11, 13, 15, 24, 30]
     * - data: ['p','l','e','a','s','e','a','l','l','o','w','m','e','t','o','i','n','t','r','o','d','u','c','e','m','y','s','e','l','f']
     * 
     * @tparam T the type of the data stored in the data buffer, not its byte representation.
     * @tparam R the reference type to the data. This type is different from the reference type of the layout,
     * which behaves like std::optional<R>.
     * @tparam CR the const reference type to the data. This type is different from the const reference of the layout,
     * which behaves like std::optional<CR>.
     * @tparam OT type of the offset values. Must be std::int64_t or std::int32_t.
     */
    template <class T, class R, class CR, layout_offset OT = std::int64_t>
    class variable_size_binary_layout
    {
    public:

        using self_type = variable_size_binary_layout<T, R, CR, OT>;
        using inner_value_type = T;
        using inner_reference = R;
        using inner_const_reference = CR;
        using bitmap_type = array_data::bitmap_type;
        using bitmap_const_reference = typename bitmap_type::const_reference;
        using value_type = std::optional<inner_value_type>;
        using const_reference = const_reference_proxy<self_type>;
        using size_type = std::size_t;
        using iterator_tag = std::contiguous_iterator_tag;

        /**
         * These types have to be public to be accessible when
         * instantiating const_value_iterator for checking the
         * requirements of subrange.
         */
        using data_type = typename T::value_type;
        using offset_iterator = OT*;
        using const_offset_iterator = const OT*;
        using data_iterator = data_type*;
        using const_data_iterator = const data_type*;
        
        using const_value_iterator = vs_binary_value_iterator<self_type, true>;
        using const_bitmap_iterator = array_data::bitmap_type::const_iterator;
        using const_iterator = layout_iterator<self_type, true>;

        // TODO: required by layout_iterator, replace them with the right types
        // when assignment for data in a variable size bienary layout is implemented
        // and implement non const overloads of `values` and `bitmap`
        using value_iterator = const_value_iterator;
        using bitmap_iterator = const_bitmap_iterator;
        // TODO: uncomment the following line and implement the non const overloads
        // of `begin` and `end`
        // using iterator = layout_iterator<self_type, false>;

        using const_bitmap_range = std::ranges::subrange<const_bitmap_iterator>;
        using const_value_range = std::ranges::subrange<const_value_iterator>;

        explicit variable_size_binary_layout(array_data data);

        size_type size() const;
        const_reference operator[](size_type i) const;

        const_iterator cbegin() const;
        const_iterator cend() const;

        const_value_range values() const;
        const_bitmap_range bitmap() const;

    private:

        const_value_iterator value_cbegin() const;
        const_value_iterator value_cend() const;

        const_bitmap_iterator bitmap_cbegin() const;
        const_bitmap_iterator bitmap_cend() const;

        bool has_value(size_type i) const;
        inner_const_reference value(size_type i) const;

        const_offset_iterator offset(size_type i) const;
        const_offset_iterator offset_end() const;
        const_data_iterator data(size_type i) const;

        array_data m_data;

        friend class const_reference_proxy<self_type>;
        friend class vs_binary_value_iterator<self_type, true>;
    };

    /*******************************************
     * vs_binary_value_iterator implementation *
     *******************************************/

    template <class L, bool is_const>
    vs_binary_value_iterator<L, is_const>::vs_binary_value_iterator(
        offset_iterator offset_it,
        data_iterator data_begin
    )
        : m_offset_it(offset_it)
        , m_data_begin(data_begin)
    {
    }

    template <class L, bool is_const>
    auto vs_binary_value_iterator<L, is_const>::dereference() const -> reference
    {
        return reference(m_data_begin + *m_offset_it, m_data_begin + *(m_offset_it + 1));
    }

    template <class L, bool is_const>
    void vs_binary_value_iterator<L, is_const>::increment()
    {
        ++m_offset_it;
    }

    template <class L, bool is_const>
    void vs_binary_value_iterator<L, is_const>::decrement()
    {
        --m_offset_it;
    }

    template <class L, bool is_const>
    void vs_binary_value_iterator<L, is_const>::advance(difference_type n)
    {
        m_offset_it += n;
    }

    template <class L, bool is_const>
    auto vs_binary_value_iterator<L, is_const>::distance_to(const self_type& rhs) const -> difference_type
    {
        return rhs.m_offset_it - m_offset_it;
    }

    template <class L, bool is_const>
    bool vs_binary_value_iterator<L, is_const>::equal(const self_type& rhs) const
    {
        return m_offset_it == rhs.m_offset_it;
    }

    template <class L, bool is_const>
    bool vs_binary_value_iterator<L, is_const>::less_than(const self_type& rhs) const
    {
        return m_offset_it < rhs.m_offset_it;
    }

    /**********************************************
     * variable_size_binary_layout implementation *
     **********************************************/

    template <class T, class R, class CR, layout_offset OT>
    variable_size_binary_layout<T, R, CR, OT>::variable_size_binary_layout(array_data data)
        : m_data(std::move(data))
    {
        assert(m_data.buffers.size() == 2u);
        //TODO: templatize back and front in buffer and uncomment the following line
        //assert(m_data.buffers[0].size() == 0u || m_data.buffers[0].back() == m_data.buffers[1].size());
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::size() const -> size_type
    {
        assert(m_data.offset <= m_data.length);
        return static_cast<size_type>(m_data.length - m_data.offset);
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::operator[](size_type i) const -> const_reference
    {
        assert(i < size());
        return const_reference(value(i), has_value(i));
    }
    
    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::cbegin() const -> const_iterator
    {
        return const_iterator(value_cbegin(), bitmap_cbegin());
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::cend() const -> const_iterator
    {
        return const_iterator(value_cend(), bitmap_cend());
    }
    
    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap() const -> const_bitmap_range
    {
        return std::ranges::subrange(bitmap_cbegin(), bitmap_cend());
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::values() const -> const_value_range
    {
        return std::ranges::subrange(value_cbegin(), value_cend());
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap_cbegin() const -> const_bitmap_iterator
    {
        return m_data.bitmap.cbegin() + m_data.offset;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap_cend() const -> const_bitmap_iterator
    {
        return m_data.bitmap.cend();
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value_cbegin() const -> const_value_iterator
    {
        return const_value_iterator(offset(0u), data(0u));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value_cend() const -> const_value_iterator
    {
        return const_value_iterator(offset_end(), data(0u));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::has_value(size_type i) const -> bool
    {
        return m_data.bitmap.test(i + m_data.offset);
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value(size_type i) const -> inner_const_reference
    {
        return inner_const_reference(data(*offset(i)), data(*offset(i+1)));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::offset(size_type i) const -> const_offset_iterator
    {
        assert(!m_data.buffers.empty());
        return m_data.buffers[0].template data<OT>() + m_data.offset + i;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::offset_end() const -> const_offset_iterator
    {
        assert(!m_data.buffers.empty());
        return m_data.buffers[0].template data<OT>() + m_data.length;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::data(size_type i) const -> const_data_iterator
    {
        assert(!m_data.buffers.empty());
        return m_data.buffers[1].template data<data_type>() + i;
    }
}

