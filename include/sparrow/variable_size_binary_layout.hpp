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

#include <functional>
#include <ranges>

#include "sparrow/array_data.hpp"
#include "sparrow/contracts.hpp"
#include "sparrow/iterator.hpp"
#include "sparrow/mp_utils.hpp"

namespace sparrow
{
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
    class vs_binary_value_iterator : public iterator_base<
                                         vs_binary_value_iterator<L, is_const>,
                                         mpl::constify_t<typename L::inner_value_type, is_const>,
                                         std::contiguous_iterator_tag,
                                         impl::get_inner_reference_t<L, is_const>>
    {
    public:

        using self_type = vs_binary_value_iterator<L, is_const>;
        using base_type = iterator_base<
            self_type,
            mpl::constify_t<typename L::inner_value_type, is_const>,
            std::contiguous_iterator_tag,
            impl::get_inner_reference_t<L, is_const>>;
        using reference = typename base_type::reference;
        using difference_type = typename base_type::difference_type;
        using size_type = typename L::size_type;
        using layout_type = mpl::constify_t<L, is_const>;

        vs_binary_value_iterator() noexcept = default;
        vs_binary_value_iterator(layout_type* layout, size_type index);

    private:

        reference dereference() const;
        void increment();
        void decrement();
        void advance(difference_type n);
        difference_type distance_to(const self_type& rhs) const;
        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

        layout_type* p_layout = nullptr;
        difference_type m_index;

        friend class iterator_access;
    };

    /**
     * @class vs_binary_reference
     *
     * @brief Implementation of reference to inner type
     * used for layout L
     *
     * @tparam L the layout type
     */
    template <class L>
    class vs_binary_reference
    {
    public:

        using self_type = vs_binary_reference<L>;
        using size_type = typename L::size_type;
        using offset_type = typename L::offset_type;
        using buffer_type = array_data::buffer_type;
        using difference_type = std::ptrdiff_t;

        vs_binary_reference(const vs_binary_reference&) = default;
        vs_binary_reference(vs_binary_reference&&) = default;

        vs_binary_reference(L* layout, size_type index);

//         template <class N = std::string>
        self_type& operator=(std::string&& new_inner_val);

        template <class N = std::string>
        bool operator==(const N& rhs) const;

    private:

        L* p_layout = nullptr;
        size_type m_index;
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
     * - offset (i.e data_ref().buffers[0]: stored in the index `0` of array_data buffers):
     * [0, 6, 11, 13, 15, 24, 30]
     * - data (i.e data_ref().buffers[1]: stored in the index `1` of array_data buffers):
     * ['p','l','e','a','s','e','a','l','l','o','w','m','e','t','o','i','n','t','r','o','d','u','c','e','m','y','s','e','l','f']
     *
     * @tparam T the type of the data stored in the data buffer, not its byte representation.
     * @tparam R the reference type to the data. This type is different from the reference type of the layout,
     * which behaves like std::optional<R>.
     * @tparam CR the const reference type to the data. This type is different from the const reference of the
     * layout, which behaves like std::optional<CR>.
     * @tparam OT type of the offset values. Must be std::int64_t or std::int32_t.
     */
    template <class T, class R, class CR, layout_offset OT = std::int64_t>
    class variable_size_binary_layout
    {
    public:

        using self_type = variable_size_binary_layout<T, R, CR, OT>;
        using inner_value_type = T;
        using offset_type = OT;
        using inner_reference = vs_binary_reference<self_type>;
        using inner_const_reference = CR;
        using bitmap_type = array_data::bitmap_type;
        using bitmap_reference = typename bitmap_type::reference;
        using bitmap_const_reference = typename bitmap_type::const_reference;
        using value_type = std::optional<inner_value_type>;
        using reference = reference_proxy<self_type>;
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

        using value_iterator = vs_binary_value_iterator<self_type, false>;
        using bitmap_iterator = array_data::bitmap_type::iterator;

        using iterator = layout_iterator<self_type, false>;
        using const_iterator = layout_iterator<self_type, true>;       

        using const_value_range = std::ranges::subrange<const_value_iterator>;
        using const_bitmap_range = std::ranges::subrange<const_bitmap_iterator>;

        using value_range = std::ranges::subrange<value_iterator>;
        using bitmap_range = std::ranges::subrange<bitmap_iterator>;

        explicit variable_size_binary_layout(array_data& data);
        void rebind_data(array_data& data);

        variable_size_binary_layout(const self_type&) = delete;
        self_type& operator=(const self_type&) = delete;
        variable_size_binary_layout(self_type&&) = delete;
        self_type& operator=(self_type&&) = delete;

        size_type size() const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        iterator begin();
        iterator end();

        const_iterator cbegin() const;
        const_iterator cend() const;

        const_value_range values() const;
        const_bitmap_range bitmap() const;

    private:

        value_iterator value_begin();
        value_iterator value_end();

        const_value_iterator value_cbegin() const;
        const_value_iterator value_cend() const;

        bitmap_iterator bitmap_begin();
        bitmap_iterator bitmap_end();

        const_bitmap_iterator bitmap_cbegin() const;
        const_bitmap_iterator bitmap_cend() const;

        bitmap_reference has_value(size_type i);
        bitmap_const_reference has_value(size_type i) const;

        inner_reference value(size_type i);
        inner_const_reference value(size_type i) const;

        const_offset_iterator offset(size_type i) const;
        const_offset_iterator offset_end() const;
        const_data_iterator data(size_type i) const;

        array_data& data_ref();
        const array_data& data_ref() const;

        std::reference_wrapper<array_data> m_data;

        friend class reference_proxy<self_type>;
        friend class const_reference_proxy<self_type>;
        friend class vs_binary_reference<self_type>;
        friend class vs_binary_value_iterator<self_type, true>;
    };

    /*******************************************
     * vs_binary_value_iterator implementation *
     *******************************************/
    template <class L, bool is_const>
    vs_binary_value_iterator<L, is_const>::vs_binary_value_iterator(layout_type* layout, size_type index)
        : p_layout(layout)
        , m_index(static_cast<difference_type>(index))
    {
    }

    template <class L, bool is_const>
    auto vs_binary_value_iterator<L, is_const>::dereference() const -> reference
    {
        if constexpr (is_const)
        {
            return p_layout->value(static_cast<size_type>(m_index));
        }
        else
        {
            return reference(p_layout, static_cast<size_type>(m_index));
        }
    }

    template <class L, bool is_const>
    void vs_binary_value_iterator<L, is_const>::increment()
    {
        ++m_index;
    }

    template <class L, bool is_const>
    void vs_binary_value_iterator<L, is_const>::decrement()
    {
        --m_index;
    }

    template <class L, bool is_const>
    void vs_binary_value_iterator<L, is_const>::advance(difference_type n)
    {
        m_index += n;
    }

    template <class L, bool is_const>
    auto vs_binary_value_iterator<L, is_const>::distance_to(const self_type& rhs) const -> difference_type
    {
        return rhs.m_index - m_index;
    }

    template <class L, bool is_const>
    bool vs_binary_value_iterator<L, is_const>::equal(const self_type& rhs) const
    {
        return p_layout == rhs.p_layout && m_index == rhs.m_index;
    }

    template <class L, bool is_const>
    bool vs_binary_value_iterator<L, is_const>::less_than(const self_type& rhs) const
    {
        return p_layout == rhs.m_layout && m_index < rhs.m_index;
    }

    /**************************************
     * vs_binary_reference implementation *
     **************************************/

    template <class L>
//     template <class N>
    auto vs_binary_reference<L>::operator=(std::string&& new_inner_val) -> self_type&
    {
            buffer_type& offset_buffer = p_layout->data_ref().buffers[0];
            buffer_type& data_buffer = p_layout->data_ref().buffers[1];
            const auto layout_data_length = p_layout->data_ref().length;

            auto initial_value_length = static_cast<size_type>(offset_buffer.template data<offset_type>()[m_index + 1] - offset_buffer.template data<offset_type>()[m_index]);
            auto new_value_length = new_inner_val.size();
            if (new_value_length > initial_value_length)
            {
                auto shift_val = new_value_length - initial_value_length;
                // Allocate tmp buffer for data
                buffer_type tmp_data_buf;
                tmp_data_buf.resize(data_buffer.size() + shift_val);
                // Copy initial elements
                std::copy(data_buffer.cbegin(), data_buffer.cbegin() + offset_buffer.template data<offset_type>()[m_index], tmp_data_buf.begin());
                std::string str(tmp_data_buf.begin(), tmp_data_buf.end());
                std::cout << "tmp after 1st copy: " << str <<  std::endl;
                // Copy new_inner_val
                std::copy(new_inner_val.cbegin(), new_inner_val.cend(), tmp_data_buf.begin() + offset_buffer.template data<offset_type>()[m_index]);
                std::string str_new(tmp_data_buf.begin(), tmp_data_buf.end());
                std::cout << "tmp after new val copy: " << str_new <<  std::endl;
                // Copy the elements left
                std::copy(data_buffer.cbegin() + offset_buffer.template data<offset_type>()[m_index + 1], data_buffer.cend(), tmp_data_buf.begin() + offset_buffer.template data<offset_type>()[m_index] + static_cast<difference_type>(new_value_length));
                std::string str_end(tmp_data_buf.begin(), tmp_data_buf.end());
                std::cout << "tmp after end: " << str_end <<  std::endl;
                std::swap(data_buffer, tmp_data_buf);
                // Update offsets
                std::cout << "lay size: " << p_layout->size() <<  " buffer data length: " << p_layout->data_ref().length << std::endl;
                std::for_each(offset_buffer.template data<offset_type>() + static_cast<difference_type>(m_index) + 1, offset_buffer.template data<offset_type>() + layout_data_length + 1, [shift_val](auto& offset){ std::cout << " offset before " << offset << std::endl; offset += static_cast<offset_type>(shift_val); });
            }
            else
            {
                std::copy(new_inner_val.cbegin(), new_inner_val.cend(), data_buffer.begin() + offset_buffer.template data<offset_type>()[m_index]);
                if (new_value_length < initial_value_length)
                {
                    std::size_t shift_val = initial_value_length - new_value_length;
                    // Shift values
                    std::copy( data_buffer.cbegin() + offset_buffer.template data<offset_type>()[m_index + 1], data_buffer.cend(), data_buffer.begin() + offset_buffer.template data<offset_type>()[m_index] + static_cast<difference_type>(new_value_length));
                    // Update offsets 
                    std::for_each(offset_buffer.template data<offset_type>() + static_cast<difference_type>(m_index) + 1, offset_buffer.template data<offset_type>() + layout_data_length + 1, [shift_val](auto& offset){ std::cout << " offset before " << offset << std::endl; offset -= static_cast<offset_type>(shift_val); });
                }
            }
            return *this;
    }

    template <class L>
    template <class N>
    bool vs_binary_reference<L>::operator==(const N& rhs) const
    {
        buffer_type& offset_buffer = p_layout->data_ref().buffers[0];
        buffer_type& data_buffer = p_layout->data_ref().buffers[1];
        std::string str_complete(data_buffer.begin(), data_buffer.end());
        std::cout << "data_buffer: " << str_complete <<  std::endl;

        std::string str(rhs.begin(), rhs.end());
        std::cout << "rhs: " << str <<  std::endl;
        std::cout << "m_index: " << offset_buffer.template data<offset_type>()[m_index] << " and " << offset_buffer.template data<offset_type>()[m_index+1]  <<  std::endl;

        std::string str_zero(data_buffer.begin()+ offset_buffer.template data<offset_type>()[m_index], data_buffer.begin() + offset_buffer.template data<offset_type>()[m_index + 1]);
        std::cout << "data buff to compare: " << str_zero <<  std::endl;

        return std::equal(rhs.cbegin(), rhs.cend(), data_buffer.cbegin() + offset_buffer.template data<offset_type>()[m_index], data_buffer.cbegin() + offset_buffer.template data<offset_type>()[m_index + 1]);
    }

    template <class L>
    vs_binary_reference<L>::vs_binary_reference(L* layout, size_type index)
        : p_layout(layout)
        , m_index(index)
    {
    }

    /**********************************************
     * variable_size_binary_layout implementation *
     **********************************************/

    template <class T, class R, class CR, layout_offset OT>
    variable_size_binary_layout<T, R, CR, OT>::variable_size_binary_layout(array_data& data)
        : m_data(data)
    {
        SPARROW_ASSERT_TRUE(data_ref().buffers.size() == 2u);
        // TODO: templatize back and front in buffer and uncomment the following line
        // SPARROW_ASSERT_TRUE(data_ref().buffers[0].size() == 0u || data_ref().buffers[0].back() ==
        // data_ref().buffers[1].size());
    }

    template <class T, class R, class CR, layout_offset OT>
    void variable_size_binary_layout<T, R, CR, OT>::rebind_data(array_data& data)
    {
        m_data = data;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::size() const -> size_type
    {
        SPARROW_ASSERT_TRUE(data_ref().offset <= data_ref().length);
        return static_cast<size_type>(data_ref().length - data_ref().offset);
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::operator[](size_type i) -> reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return reference(value(i), has_value(i));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::operator[](size_type i) const -> const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        return const_reference(value(i), has_value(i));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::begin() -> iterator
    {
        return iterator(value_begin(), bitmap_begin());
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::end() -> iterator
    {
        return iterator(value_end(), bitmap_end());
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
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap_begin() -> bitmap_iterator
    {
        return data_ref().bitmap.begin() + data_ref().offset;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap_end() -> bitmap_iterator
    {
        return data_ref().bitmap.end();
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap_cbegin() const -> const_bitmap_iterator
    {
        return data_ref().bitmap.cbegin() + data_ref().offset;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::bitmap_cend() const -> const_bitmap_iterator
    {
        return data_ref().bitmap.cend();
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value_begin() -> value_iterator
    {
        return value_iterator(offset(0u), data(0u));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value_end() -> value_iterator
    {
        return value_iterator(offset_end(), data(0u));
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value_cbegin() const -> const_value_iterator
    {
        return const_value_iterator(this, 0u);
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value_cend() const -> const_value_iterator
    {
        return const_value_iterator(this, size());
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::has_value(size_type i) -> bitmap_reference
    {
        SPARROW_ASSERT_TRUE(data_ref().offset >= 0);
        const size_type pos = i + static_cast<size_type>(data_ref().offset);
        return data_ref().bitmap[pos];
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::has_value(size_type i) const -> bitmap_const_reference
    {
        SPARROW_ASSERT_TRUE(data_ref().offset >= 0);
        const size_type pos = i + static_cast<size_type>(data_ref().offset);
        return data_ref().bitmap[pos];
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value(size_type i) -> inner_reference
    {
        const size_type pos = i + static_cast<size_type>(data_ref().offset);
        return inner_reference(this, pos);
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::value(size_type i) const -> inner_const_reference
    {
        const long long offset_i = *offset(i);
        SPARROW_ASSERT_TRUE(offset_i >= 0);
        const long long offset_next = *offset(i + 1);
        SPARROW_ASSERT_TRUE(offset_next >= 0);
        const const_data_iterator pointer1 = data(static_cast<size_type>(offset_i));
        const const_data_iterator pointer2 = data(static_cast<size_type>(offset_next));
        return inner_const_reference(pointer1, pointer2);
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::offset(size_type i) const -> const_offset_iterator
    {
        SPARROW_ASSERT_FALSE(data_ref().buffers.empty());
        return data_ref().buffers[0].template data<OT>() + data_ref().offset + i;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::offset_end() const -> const_offset_iterator
    {
        SPARROW_ASSERT_FALSE(data_ref().buffers.empty());
        return data_ref().buffers[0].template data<OT>() + data_ref().length;
    }

    template <class T, class R, class CR, layout_offset OT>
    auto variable_size_binary_layout<T, R, CR, OT>::data(size_type i) const -> const_data_iterator
    {
        SPARROW_ASSERT_FALSE(data_ref().buffers.empty());
        return data_ref().buffers[1].template data<data_type>() + i;
    }

    template <class T, class R, class CR, layout_offset OT>
    array_data& variable_size_binary_layout<T, R, CR, OT>::data_ref()
    {
        return m_data.get();
    }

    template <class T, class R, class CR, layout_offset OT>
    const array_data& variable_size_binary_layout<T, R, CR, OT>::data_ref() const
    {
        return m_data.get();
    }
}
