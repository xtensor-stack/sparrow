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

#include "sparrow/array_data.hpp"
#include "sparrow/contracts.hpp"
#include "sparrow/fixed_size_layout.hpp"
#include "sparrow/iterator.hpp"
#include "sparrow/mp_utils.hpp"

namespace sparrow
{
    /**
     * @class dictionary_value_iterator
     *
     * @brief Iterator over the data values of a dictionary layout.
     *
     * @tparam IL the layout type of the indexes.
     * @tparam SL the layout type of the dictionary.
     * @tparam is_const a boolean flag specifying whether this iterator is const.
     */
    template <class IL, class SL, bool is_const>
    class dictionary_value_iterator : public iterator_base<
                                          dictionary_value_iterator<IL, SL, is_const>,
                                          typename SL::value_type,
                                          std::random_access_iterator_tag,
                                          typename SL::const_reference>
    {
    public:

        using self_type = dictionary_value_iterator<IL, SL, is_const>;
        using base_type = iterator_base<self_type, typename SL::value_type, std::random_access_iterator_tag, typename SL::const_reference>;
        using reference = typename base_type::reference;
        using difference_type = typename base_type::difference_type;

        using index_iterator = std::conditional_t<is_const, typename IL::const_value_iterator, typename IL::value_iterator>;
        using sub_layout = mpl::constify_t<SL, is_const>;
        using sub_layout_reference = sub_layout&;

        // `dictionary_value_iterator` needs to be default constructible
        // to satisfy `dictionary_encoded_layout::const_value_range`'s
        // constraints.
        dictionary_value_iterator() noexcept = default;
        dictionary_value_iterator(index_iterator index_it, sub_layout_reference sub_layout_reference);

    private:

        reference dereference() const;
        void increment();
        void decrement();
        void advance(difference_type n);
        difference_type distance_to(const self_type& rhs) const;
        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

        index_iterator m_index_it;
        // Use std::optional because of default constructor.
        std::optional<std::reference_wrapper<sub_layout>> m_sub_layout_reference;

        friend class iterator_access;
    };

    /*
     * @class dictionary_encoded_layout
     *
     * @brief Layout for arrays containing many duplicated values.
     *
     * Dictionary encoding is a data representation technique to represent values by
     * integers referencing a dictionary usually consisting of unique values. It can
     * be effective when you have data with many repeated values.
     *
     * Example:
     *
     * data VarBinary (dictionary-encoded)
     *   index_type: Int32
     *   values: [0, 1, 3, 1, 4, 2]
     *
     * dictionary
     *   type: VarBinary
     *   values: ['foo', 'bar', 'baz', 'foo', null]
     *
     * Traversing the values will give you the following:
     *  'foo', 'bar', 'foo', 'bar', null, 'baz'
     *
     * @tparam IT the type of the index. Must be an integral.
     * @tparam SL the layout type of the dictionary.
     * @tparam OT type of the offset values. Must be std::int64_t or std::int32_t.
     */
    template <std::integral IT, class SL, layout_offset OT = std::int64_t>
    class dictionary_encoded_layout
    {
    public:

        using self_type = dictionary_encoded_layout<IT, SL, OT>;
        using index_type = IT;
        using inner_value_type = SL::inner_value_type;
        using sub_layout = SL;
        using inner_reference = reference_proxy<SL>;
        using inner_const_reference = const_reference_proxy<SL>;
        using bitmap_type = array_data::bitmap_type;
        using bitmap_const_reference = bitmap_type::const_reference;
        using value_type = SL::value_type;
        using reference = reference_proxy<SL>;
        using const_reference = const_reference_proxy<SL>;
        using size_type = std::size_t;
        using indexes_layout = fixed_size_layout<IT>;
        using iterator_tag = std::random_access_iterator_tag;

        /**
         * These types have to be public to be accessible when
         * instantiating const_value_iterator for checking the
         * requirements of subrange.
         */
        using data_type = IT;

        using offset_iterator = OT*;
        using const_offset_iterator = const OT*;

        using data_iterator = data_type*;
        using const_data_iterator = const data_type*;

        using iterator = layout_iterator<self_type, false>;
        using const_iterator = layout_iterator<self_type, true>;

        using bitmap_iterator = indexes_layout::bitmap_iterator;
        using const_bitmap_iterator = indexes_layout::const_bitmap_iterator;
        using const_bitmap_range = indexes_layout::const_bitmap_range;

        using value_iterator = dictionary_value_iterator<indexes_layout, sub_layout, false>;
        using const_value_iterator = dictionary_value_iterator<indexes_layout, sub_layout, true>;
        using const_value_range = std::ranges::subrange<const_value_iterator, const_value_iterator>;

        explicit dictionary_encoded_layout(array_data& data);
        void rebind_data(array_data& data);

        dictionary_encoded_layout(const dictionary_encoded_layout&) = delete;
        dictionary_encoded_layout& operator=(const dictionary_encoded_layout&) = delete;
        dictionary_encoded_layout(dictionary_encoded_layout&&) = delete;
        dictionary_encoded_layout& operator=(dictionary_encoded_layout&&) = delete;

        size_type size() const;
        const_reference operator[](size_type i) const;

        const_iterator cbegin() const;
        const_iterator cend() const;

        const_bitmap_range bitmap() const;
        const_value_range values() const;

    private:

        indexes_layout& get_indexes_layout();
        const indexes_layout& get_indexes_layout() const;

        sub_layout& get_sub_layout();
        const sub_layout& get_sub_layout() const;

        const_value_iterator value_cbegin() const;
        const_value_iterator value_cend() const;

        inner_const_reference value(size_type i) const;

        const_offset_iterator offset(size_type i) const;
        const_offset_iterator offset_end() const;
        const_data_iterator data(size_type i) const;

        std::unique_ptr<indexes_layout> m_indexes_layout;
        std::unique_ptr<sub_layout> m_sub_layout;

        static const const_reference& dummy_const_reference()
        {
            static const typename sub_layout::inner_value_type dummy_inner_value;
            static const typename sub_layout::bitmap_type dummy_bitmap(1, false);
            static const const_reference instance(dummy_inner_value, dummy_bitmap[0]);
            return instance;
        }

        friend class const_reference_proxy<self_type>;
        friend class dictionary_value_iterator<indexes_layout, sub_layout, true>;
    };

    /*******************************************
     * vs_binary_value_iterator implementation *
     *******************************************/

    template <class L, class SL, bool is_const>
    dictionary_value_iterator<L, SL, is_const>::dictionary_value_iterator(
        index_iterator index_it,
        sub_layout_reference sub_layout_ref
    )
        : m_index_it(index_it)
        , m_sub_layout_reference(sub_layout_ref)
    {
    }

    template <class IL, class SL, bool is_const>
    auto dictionary_value_iterator<IL, SL, is_const>::dereference() const -> reference
    {
        SPARROW_ASSERT_TRUE(m_sub_layout_reference.has_value());
        return (*m_sub_layout_reference).get()[*m_index_it];
    }

    template <class IL, class SL, bool is_const>
    void dictionary_value_iterator<IL, SL, is_const>::increment()
    {
        ++m_index_it;
    }

    template <class IL, class SL, bool is_const>
    void dictionary_value_iterator<IL, SL, is_const>::decrement()
    {
        --m_index_it;
    }

    template <class IL, class SL, bool is_const>
    void dictionary_value_iterator<IL, SL, is_const>::advance(difference_type n)
    {
        m_index_it += n;
    }

    template <class IL, class SL, bool is_const>
    auto dictionary_value_iterator<IL, SL, is_const>::distance_to(const self_type& rhs) const -> difference_type
    {
        m_index_it.distance_to(rhs.m_index_it);
    }

    template <class IL, class SL, bool is_const>
    bool dictionary_value_iterator<IL, SL, is_const>::equal(const self_type& rhs) const
    {
        return m_index_it == rhs.m_index_it;
    }

    template <class IL, class SL, bool is_const>
    bool dictionary_value_iterator<IL, SL, is_const>::less_than(const self_type& rhs) const
    {
        return m_index_it < rhs.m_index_it;
    }

    /**********************************************
     * dictionary_encoded_layout implementation *
     **********************************************/

    template <std::integral T, class SL, layout_offset OT>
    dictionary_encoded_layout<T, SL, OT>::dictionary_encoded_layout(array_data& data)
    {
        SPARROW_ASSERT_TRUE(data.dictionary);
        m_sub_layout = std::make_unique<SL>(*data.dictionary);
        m_indexes_layout = std::make_unique<indexes_layout>(data);
    }

    template <std::integral T, class SL, layout_offset OT>
    void dictionary_encoded_layout<T, SL, OT>::rebind_data(array_data& data)
    {
        m_sub_layout->rebind_data(*data.dictionary);
        m_indexes_layout->rebind_data(data);
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::size() const -> size_type
    {
        return m_indexes_layout->size();
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::operator[](size_type i) const -> const_reference
    {
        SPARROW_ASSERT_TRUE(i < size());
        const auto index = get_indexes_layout()[i];
        if (index.has_value())
        {
            return get_sub_layout()[index.value()];
        }
        else
        {
            return dummy_const_reference();
        }
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::bitmap() const -> const_bitmap_range
    {
        return get_indexes_layout().bitmap();
    }

    template <std::integral T, class SL, layout_offset OT>
    typename dictionary_encoded_layout<T, SL, OT>::indexes_layout&
    dictionary_encoded_layout<T, SL, OT>::get_indexes_layout()
    {
        return *(m_indexes_layout.get());
    }

    template <std::integral T, class SL, layout_offset OT>
    const typename dictionary_encoded_layout<T, SL, OT>::indexes_layout&
    dictionary_encoded_layout<T, SL, OT>::get_indexes_layout() const
    {
        return *const_cast<const indexes_layout*>(m_indexes_layout.get());
    }

    template <std::integral T, class SL, layout_offset OT>
    typename dictionary_encoded_layout<T, SL, OT>::sub_layout&
    dictionary_encoded_layout<T, SL, OT>::get_sub_layout()
    {
        return *(m_sub_layout.get());
    }

    template <std::integral T, class SL, layout_offset OT>
    const typename dictionary_encoded_layout<T, SL, OT>::sub_layout&
    dictionary_encoded_layout<T, SL, OT>::get_sub_layout() const
    {
        return *const_cast<const sub_layout*>(m_sub_layout.get());
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::cbegin() const -> const_iterator
    {
        return const_iterator(value_cbegin(), get_indexes_layout().bitmap().begin());
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::cend() const -> const_iterator
    {
        return const_iterator(value_cend(), get_indexes_layout().bitmap().end());
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::value_cbegin() const -> const_value_iterator
    {
        return const_value_iterator(get_indexes_layout().values().begin(), get_sub_layout());
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::value_cend() const -> const_value_iterator
    {
        return const_value_iterator(get_indexes_layout().values().end(), get_sub_layout());
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::values() const -> const_value_range
    {
        return const_value_range(value_cbegin(), value_cend());
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::value(size_type i) const -> inner_const_reference
    {
        return inner_const_reference(data(*offset(i)), data(*offset(i + 1)));
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::offset(size_type i) const -> const_offset_iterator
    {
        return m_indexes_layout->offset(i);
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::offset_end() const -> const_offset_iterator
    {
        return m_indexes_layout->offset_end();
    }

    template <std::integral T, class SL, layout_offset OT>
    auto dictionary_encoded_layout<T, SL, OT>::data(size_type i) const -> const_data_iterator
    {
        return m_sub_layout->data(i);
    }
}  // namespace sparrow
