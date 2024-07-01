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

#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "sparrow/allocator.hpp"
#include "sparrow/arrow_array_schema_utils.hpp"
#include "sparrow/buffer.hpp"
#include "sparrow/c_interface.hpp"

namespace sparrow
{
    struct arrow_array_custom_deleter
    {
        void operator()(ArrowArray* array) const
        {
            if (array->release != nullptr)
            {
                array->release(array);
            }
            delete array;
        }
    };

    using arrow_array_unique_ptr = std::unique_ptr<ArrowArray, arrow_array_custom_deleter>;

    using arrow_array_shared_ptr = std::shared_ptr<ArrowArray>;  // TODO: Create a class for shared_ptr
                                                                 // ArrowArray which always uses the custom
                                                                 // deleter

    /**
     * Struct representing private data for ArrowArray.
     *
     * This struct holds the private data for ArrowArray, including buffers, children, and dictionary.
     * It is used in the Sparrow library.
     *
     * @tparam BufferType The type of the buffers.
     * @tparam Allocator An allocator for BufferType.
     */
    template <class BufferType, template <typename> class Allocator = std::allocator>
        requires sparrow::allocator<Allocator<BufferType>>
    struct arrow_array_private_data
    {
        template <typename B, typename C, typename D>
        explicit arrow_array_private_data(B buffers, C children, D dictionary)
            : m_buffers_raw_ptr_ptr(get_buffers<BufferType>(std::move(buffers), m_buffers_raw_ptr_vec, m_buffers))
            , m_children_raw_ptr_ptr(
                  get_children<ArrowArray>(std::move(children), m_children_raw_ptr_vec, m_children)
              )
            , m_dictionary_raw_ptr(get_dictionary<ArrowArray>(std::move(dictionary), m_dictionary))
        {
        }

        [[nodiscard]] const BufferType** buffers() noexcept
        {
            return const_cast<const BufferType**>(m_buffers_raw_ptr_ptr);
        }

        [[nodiscard]] ArrowArray** children() noexcept
        {
            return m_children_raw_ptr_ptr;
        }

        [[nodiscard]] ArrowArray* dictionary() const noexcept
        {
            return m_dictionary_raw_ptr;
        }

    private:

        using buffer_allocator = Allocator<BufferType>;
        using buffers_allocator = Allocator<buffer<BufferType>>;

        buffer_allocator m_buffer_allocator = buffer_allocator();
        buffers_allocator m_buffers_allocator = buffers_allocator();

        std::any m_buffers;
        std::vector<BufferType*> m_buffers_raw_ptr_vec;
        BufferType** m_buffers_raw_ptr_ptr = nullptr;

        std::any m_children;
        std::vector<ArrowArray*> m_children_raw_ptr_vec;
        ArrowArray** m_children_raw_ptr_ptr = nullptr;

        std::optional<std::variant<arrow_array_unique_ptr, std::shared_ptr<ArrowArray>>> m_dictionary;
        ArrowArray* m_dictionary_raw_ptr = nullptr;
    };

    /**
     * Deletes an ArrowArray.
     *
     * @tparam T The type of the buffers of the ArrowArray.
     * @tparam Allocator The allocator for the buffers of the ArrowArray.
     * @param array The ArrowArray to delete. Should be a valid pointer to an ArrowArray. Its release callback
     * should be set.
     */
    template <typename T, template <typename> typename Allocator>
        requires sparrow::allocator<Allocator<T>>
    void delete_array(ArrowArray* array)
    {
        SPARROW_ASSERT_FALSE(array == nullptr)

        array->buffers = nullptr;
        array->n_buffers = 0;
        array->length = 0;
        array->null_count = 0;
        array->offset = 0;
        array->n_children = 0;
        array->children = nullptr;
        array->dictionary = nullptr;
        if (array->private_data != nullptr)
        {
            const auto private_data = static_cast<arrow_array_private_data<T, Allocator>*>(array->private_data);
            delete private_data;
        }
        array->private_data = nullptr;
        array->release = nullptr;
    }

    arrow_array_unique_ptr default_arrow_array()
    {
        auto ptr = arrow_array_unique_ptr(new ArrowArray());
        ptr->length = 0;
        ptr->null_count = 0;
        ptr->offset = 0;
        ptr->n_buffers = 0;
        ptr->n_children = 0;
        ptr->buffers = nullptr;
        ptr->children = nullptr;
        ptr->dictionary = nullptr;
        ptr->release = nullptr;
        ptr->private_data = nullptr;
        return ptr;
    }

    // TODO: Auto deduction of T from B
    template <class T, template <typename> class Allocator, class B, class C, class D>
        requires sparrow::allocator<Allocator<T>>
                 && (std::ranges::sized_range<B> || std::is_same_v<B, std::nullptr_t>)
                 && (std::ranges::sized_range<C> || std::is_same_v<C, std::nullptr_t>)
    arrow_array_unique_ptr
        make_arrow_array(int64_t length, int64_t null_count, int64_t offset, B buffers, C children, D dictionary);

    // Creates an ArrowArray.
    //
    // @tparam T The type of the buffers of the ArrowArray
    // @tparam Allocator The allocator for the buffers of the ArrowArray.
    // @param length The logical length of the array (i.e. its number of items). Must be 0 or positive.
    // @param null_count The number of null items in the array. May be -1 if not yet computed. Must be 0 or
    // positive otherwise.
    // @param offset The logical offset inside the array (i.e. the number of items from the physical start of
    //               the buffers). Must be 0 or positive.
    // @param n_buffers The number of physical buffers backing this array. The number of buffers is a
    //                  function of the data type, as described in the Columnar format specification, except
    //                  for the the binary or utf-8 view type, which has one additional buffer compared to the
    //                  Columnar format specification (see Binary view arrays). Must be 0 or positive.
    // @param children Vector of child arrays. Children must not be null.
    // @param dictionary A pointer to the underlying array of dictionary values. Must be present if the
    //                   ArrowArray represents a dictionary-encoded array. Must be null otherwise.
    // @return The created ArrowArray.
    // TODO: Auto deduction of T from B
    template <class T, template <typename> class Allocator, class B, class C, class D>
        requires sparrow::allocator<Allocator<T>>
    arrow_array_unique_ptr make_arrow_array(
        int64_t length,
        int64_t null_count,
        int64_t offset,
        int64_t n_buffers,
        B buffers,
        int64_t n_children,
        C children,
        D dictionary
    )
    {
        SPARROW_ASSERT_TRUE(length >= 0);
        SPARROW_ASSERT_TRUE(null_count >= -1);
        SPARROW_ASSERT_TRUE(offset >= 0);
        SPARROW_ASSERT_TRUE(n_buffers >= 0);
        SPARROW_ASSERT_TRUE(n_children >= 0);

        arrow_array_unique_ptr array = default_arrow_array();

        array->length = length;
        array->null_count = null_count;
        array->offset = offset;
        array->n_buffers = n_buffers;

        array->private_data = new arrow_array_private_data<T, Allocator>(
            std::move(buffers),
            std::move(children),
            std::move(dictionary)
        );
        const auto private_data = static_cast<arrow_array_private_data<T, Allocator>*>(array->private_data);
        array->buffers = reinterpret_cast<const void**>(private_data->buffers());
        array->n_children = n_children;
        array->children = private_data->children();
        array->dictionary = private_data->dictionary();
        array->release = delete_array<T, Allocator>;
        return array;
    }

    // TODO: Auto deduction of T from B
    template <class T, template <typename> class Allocator, class B, class C, class D>
        requires sparrow::allocator<Allocator<T>>
                 && (std::ranges::sized_range<B> || std::is_same_v<B, std::nullptr_t>)
                 && (std::ranges::sized_range<C> || std::is_same_v<C, std::nullptr_t>)
    arrow_array_unique_ptr
        make_arrow_array(int64_t length, int64_t null_count, int64_t offset, B buffers, C children, D dictionary)
    {
        const int64_t buffer_count = get_size(buffers);
        const int64_t children_count = get_size(children);
        return make_arrow_array<T, Allocator, B, C, D>(
            length,
            null_count,
            offset,
            buffer_count,
            std::move(buffers),
            children_count,
            std::move(children),
            std::move(dictionary)
        );
    }

}
