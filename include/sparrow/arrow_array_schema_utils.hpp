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

#include <algorithm>
#include <any>
#include <optional>
#include <ranges>
#include <vector>

#include "sparrow/buffer.hpp"
#include "sparrow/c_interface.hpp"
#include "sparrow/mp_utils.hpp"

namespace sparrow
{
    template <typename T>
    concept any_arrow_array = std::is_same_v<T, ArrowArray> || std::is_same_v<T, ArrowSchema>;

    template <class T, std::ranges::input_range Range, template <typename> class Allocator = std::allocator>
    std::vector<T*, Allocator<T*>> to_raw_ptr_vec(Range& range)
    {
        std::vector<T*, Allocator<T*>> raw_ptr_vec;
        raw_ptr_vec.reserve(range.size());
        std::ranges::transform(
            range,
            std::back_inserter(raw_ptr_vec),
            [](auto& elem) -> T*
            {
                using Range_Element = std::ranges::range_value_t<Range>;
                if constexpr (std::is_pointer_v<Range_Element>)
                {
                    return elem;
                }
                else if constexpr (mpl::is_smart_ptr<Range_Element>)
                {
                    if constexpr (std::is_same_v<typename Range_Element::element_type, T>)
                    {
                        return elem.get();
                    }
                    else if constexpr (std::ranges::input_range<typename Range_Element::element_type>)
                    {
                        return elem.get()->data();
                    }
                }
                else if constexpr (mpl::is_type_instance_of_v<Range_Element, sparrow::buffer>)
                {
                    return elem.data();
                }
                else
                {
                    static_assert(mpl::dependent_false<Range_Element>::value, "Invalid type for range element");
                    mpl::unreachable();
                }
            }
        );
        return raw_ptr_vec;
    }

    template <class BufferType, template <typename> class Allocator = std::allocator>
        requires sparrow::allocator<Allocator<buffer<BufferType>>>
    std::vector<buffer<BufferType>, Allocator<buffer<BufferType>>> create_buffers(
        size_t buffer_size,
        size_t buffer_count,
        const Allocator<BufferType>& buffer_allocator,
        const Allocator<buffer<BufferType>>& buffers_allocator_
    )
    {
        std::vector<buffer<BufferType>, Allocator<buffer<BufferType>>> buffers(buffers_allocator_);
        buffers.reserve(buffer_count);
        for (size_t i = 0; i < buffer_count; ++i)
        {
            buffers.emplace_back(buffer_size, buffer_allocator);
        }
        return buffers;
    }

    // TODO: Create a shared pointer with custom deleter. We should not allow the use of a shared_ptr which
    // don't use our custom deleter struct arrow_array_shared_ptr : public std::shared_ptr<ArrowArray>
    // {
    //     template <typename... Args>
    //     explicit arrow_array_shared_ptr(Args&&... args)
    //         : std::shared_ptr<ArrowArray>(std::forward<Args>(args)..., arrow_array_custom_deleter())
    //     {
    //     }
    // };

    template <class T, class D, class SmartPtrOutput>
    T* get_dictionary(D dictionary, std::optional<SmartPtrOutput>& dictionary_smart_ptr)
    {
        if constexpr (std::is_same_v<D, T*> || std::is_same_v<D, std::nullptr_t>)
        {
            return dictionary;
        }
        else if constexpr (mpl::is_smart_ptr<D> && std::is_same_v<typename D::element_type, T>)
        {
            dictionary_smart_ptr = SmartPtrOutput{std::move(dictionary)};
            return std::visit(
                [](auto& dict)
                {
                    return dict.get();
                },
                *dictionary_smart_ptr
            );
        }
        else
        {
            static_assert(mpl::dependent_false<D>::value, "Invalid type for dictionary");
            mpl::unreachable();
        }
    }

    template <std::ranges::input_range Input>
        requires mpl::is_unique_ptr<std::ranges::range_value_t<Input>>
    std::vector<std::shared_ptr<typename std::ranges::range_value_t<Input>::element_type>>
    range_of_unique_ptr_to_vec_of_shared_ptr(Input& input)
    {
        using T = std::ranges::range_value_t<Input>::element_type;
        std::vector<std::shared_ptr<T>> shared_ptrs;
        shared_ptrs.reserve(std::ranges::size(input));
        std::ranges::transform(
            input,
            std::back_inserter(shared_ptrs),
            [](auto& child)
            {
                return std::shared_ptr<T>(child.release(), child.get_deleter());
            }
        );
        return shared_ptrs;
    }

    template <class T, class C>
    T** get_children(C children, std::vector<T*>& vec_ptr_output, std::any& children_output)
    {
        if constexpr (std::is_same_v<C, T**> || std::is_same_v<C, std::nullptr_t>)
        {
            return children;
        }
        else if constexpr (std::is_same_v<C, std::vector<T*>>)
        {
            vec_ptr_output = std::move(children);
            return vec_ptr_output.data();
        }
        else if constexpr (std::ranges::input_range<C>)
        {
            using Element = std::ranges::range_value_t<C>;

            // In case where `children` is a range of objects, raw pointers or shared pointers
            if constexpr (std::is_same_v<Element, T> || std::is_same_v<Element, T*>
                          || (mpl::is_shared_ptr<Element> && std::is_same_v<typename Element::element_type, T>) )
            {
                children_output = std::move(children);
                C& children_ptr = std::any_cast<C&>(children_output);
                vec_ptr_output = to_raw_ptr_vec<T>(children_ptr);
                return vec_ptr_output.data();
            }
            // In the case where `children` is a range of unique pointers, we have to transform them to shared
            // pointers
            else if constexpr (mpl::is_unique_ptr<Element> && std::is_same_v<typename Element::element_type, T>)
            {
                children_output = range_of_unique_ptr_to_vec_of_shared_ptr(children);
                auto& children_ptr = std::any_cast<std::vector<std::shared_ptr<T>>&>(children_output);
                vec_ptr_output = to_raw_ptr_vec<T>(children_ptr);
                return vec_ptr_output.data();
            }
        }
        else
        {
            static_assert(mpl::dependent_false<C>::value, "Invalid type for children");
            mpl::unreachable();
        }
    }

    template <class T, class B>
    T** get_buffers(B buffers, std::vector<T*>& vec_ptr_output, std::any& buffers_output)
    {
        if constexpr (std::is_same_v<B, T**> || std::is_same_v<B, std::nullptr_t>)
        {
            return buffers;
        }
        else if constexpr (std::is_same_v<B, std::vector<T*>>)
        {
            vec_ptr_output = std::move(buffers);
            return vec_ptr_output.data();
        }
        else if constexpr (std::ranges::input_range<B>)
        {
            using Element = std::ranges::range_value_t<B>;
            if constexpr (std::ranges::input_range<Element>)
            {
                using BufferType = std::ranges::range_value_t<Element>;
                buffers_output = std::move(buffers);
                B& buffers_ptr = std::any_cast<B&>(buffers_output);
                vec_ptr_output = to_raw_ptr_vec<BufferType>(buffers_ptr);
                return vec_ptr_output.data();
            }
            else if constexpr (std::is_same_v<Element, T*>
                               || (mpl::is_smart_ptr<Element>
                                   && std::is_same_v<std::ranges::range_value_t<typename Element::element_type>, T>) )
            {
                buffers_output = std::move(buffers);
                B& buffers_ptr = std::any_cast<B&>(buffers_output);
                vec_ptr_output = to_raw_ptr_vec<T>(buffers_ptr);
                return vec_ptr_output.data();
            }
        }
        else
        {
            static_assert(mpl::dependent_false<B>::value, "Invalid type for buffers");
            mpl::unreachable();
        }
    }

    template <class T>
    int64_t get_size(const T& value)
        requires(std::ranges::sized_range<T> || std::is_same_v<T, std::nullptr_t>)
    {
        if constexpr (std::ranges::sized_range<T>)
        {
            return static_cast<int64_t>(std::ranges::size(value));
        }
        else
        {
            return 0;
        }
    }
}
