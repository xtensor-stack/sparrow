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
#include <span>

#include "sparrow/allocator.hpp"
#include "sparrow/arrow_array_schema_utils.hpp"
#include "sparrow/c_interface.hpp"
#include "sparrow/contracts.hpp"


namespace sparrow
{
    struct arrow_schema_custom_deleter
    {
        void operator()(ArrowSchema* schema) const
        {
            if (schema->release != nullptr)
            {
                schema->release(schema);
            }
            delete schema;
        }
    };

    using arrow_schema_unique_ptr = std::unique_ptr<ArrowSchema, arrow_schema_custom_deleter>;

    /**
     * Struct representing private data for ArrowSchema.
     *
     * This struct holds the private data for ArrowArray, including format, name and metadata strings,
     * children, and dictionary. It is used in the Sparrow library.
     *
     * @tparam Allocator An allocator for char.
     */
    template <template <typename> class Allocator = std::allocator>
        requires sparrow::allocator<Allocator<char>>
    struct arrow_schema_private_data
    {
        using string_type = std::basic_string<char, std::char_traits<char>, Allocator<char>>;
        using vector_string_type = std::vector<char, Allocator<char>>;

        arrow_schema_private_data() = delete;
        arrow_schema_private_data(const arrow_schema_private_data&) = delete;
        arrow_schema_private_data(arrow_schema_private_data&&) = delete;
        arrow_schema_private_data& operator=(const arrow_schema_private_data&) = delete;
        arrow_schema_private_data& operator=(arrow_schema_private_data&&) = delete;

        template <class C, class D>
        explicit arrow_schema_private_data(
            std::string_view format_view,
            std::string_view name_view,
            const std::optional<std::span<char>>& metadata,
            C children,
            D dictionary
        )
            : m_format(format_view)
            , m_name(name_view)
            , m_metadata(
                  metadata.has_value() ? vector_string_type(metadata->begin(), metadata->end())
                                       : vector_string_type()
              )
            , m_children_raw_ptr_ptr(get_children(std::move(children), m_children_raw_ptr_vec, m_children))
            , m_dictionary_raw_ptr(get_dictionary<ArrowSchema>(std::move(dictionary), m_dictionary))
        {
        }

        ~arrow_schema_private_data();

        [[nodiscard]] const char* format() const noexcept
        {
            if (!m_format.empty())
            {
                return m_format.data();
            }
            return nullptr;
        }

        [[nodiscard]] const char* name() const noexcept
        {
            if (!m_name.empty())
            {
                return m_name.data();
            }
            return nullptr;
        }

        [[nodiscard]] const char* metadata() const noexcept
        {
            if (m_metadata.has_value() && !m_metadata->empty())
            {
                return m_metadata->data();
            }
            return nullptr;
        }

        [[nodiscard]] ArrowSchema** children() noexcept
        {
            return m_children_raw_ptr_ptr;
        }

        [[nodiscard]] ArrowSchema* dictionary() const noexcept
        {
            return m_dictionary_raw_ptr;
        }

    private:

        sparrow::any_allocator<char> string_allocator_ = Allocator<char>();
        string_type m_format;
        string_type m_name;
        std::optional<vector_string_type> m_metadata;
        std::any m_children;
        std::vector<ArrowSchema*> m_children_raw_ptr_vec;
        ArrowSchema** m_children_raw_ptr_ptr = nullptr;
        std::optional<std::variant<arrow_schema_unique_ptr, std::shared_ptr<ArrowSchema>>> m_dictionary;
        ArrowSchema* m_dictionary_raw_ptr = nullptr;
    };

    template <template <typename> class Allocator>
        requires sparrow::allocator<Allocator<char>>
    arrow_schema_private_data<Allocator>::~arrow_schema_private_data()
    {
        if (m_children.has_value())
        {
            for (auto& child : m_children_raw_ptr_vec)
            {
                SPARROW_ASSERT_TRUE(child->release == nullptr)
                if (child->release != nullptr)
                {
                    child->release(child);
                }
            }
        }

        if (m_dictionary_raw_ptr != nullptr)
        {
            SPARROW_ASSERT_TRUE(m_dictionary_raw_ptr->release == nullptr)
            if (m_dictionary_raw_ptr->release != nullptr)
            {
                m_dictionary_raw_ptr->release(m_dictionary_raw_ptr);
            }
        }
    }

    /**
     * Deletes an ArrowSchema.
     *
     * @tparam Allocator The allocator for the strings of the ArrowSchema.
     * @param schema The ArrowSchema to delete. Should be a valid pointer to an ArrowSchema. Its release
     * callback should be set.
     */
    template <template <typename> class Allocator>
        requires sparrow::allocator<Allocator<char>>
    void delete_schema(ArrowSchema* schema)
    {
        SPARROW_ASSERT_FALSE(schema == nullptr)
        SPARROW_ASSERT_TRUE(schema->release == std::addressof(delete_schema<Allocator>))

        schema->flags = 0;
        schema->n_children = 0;
        schema->children = nullptr;
        schema->dictionary = nullptr;
        schema->name = nullptr;
        schema->format = nullptr;
        schema->metadata = nullptr;
        if (schema->private_data != nullptr)
        {
            const auto private_data = static_cast<arrow_schema_private_data<Allocator>*>(schema->private_data);
            delete private_data;
        }
        schema->private_data = nullptr;
        schema->release = nullptr;
    }

    arrow_schema_unique_ptr default_arrow_schema()
    {
        auto ptr = arrow_schema_unique_ptr(new ArrowSchema());
        ptr->format = nullptr;
        ptr->name = nullptr;
        ptr->metadata = nullptr;
        ptr->flags = 0;
        ptr->n_children = 0;
        ptr->children = nullptr;
        ptr->dictionary = nullptr;
        ptr->release = nullptr;
        ptr->private_data = nullptr;
        return ptr;
    }

    /**
     * Creates an ArrowSchema.
     *
     * @tparam Allocator The allocator for the strings of the ArrowSchema.
     * @param format A mandatory, null-terminated, UTF8-encoded string describing the data type. If the data
     *               type is nested, child types are not encoded here but in the ArrowSchema.children
     *               structures.
     * @param name An optional (nullptr), null-terminated, UTF8-encoded string of the field or array name.
     *             This is mainly used to reconstruct child fields of nested types.
     * @param metadata An optional (nullptr), binary string describing the type’s metadata. If the data type
     *                 is nested, the metadata for child types are not encoded here but in the
     * ArrowSchema.children structures.
     * @param flags A bitfield of flags enriching the type description. Its value is computed by OR’ing
     *              together the flag values.
     * @param children Vector of unique pointer of ArrowSchema. Children must not be null.
     * @param dictionary Optional. A pointer to the type of dictionary values. Must be present if the
     *                   ArrowSchema represents a dictionary-encoded type. Must be nullptr otherwise.
     * @return The created ArrowSchema.
     */
    template <template <typename> class Allocator, class C, class D>
        requires sparrow::allocator<Allocator<char>>
    arrow_schema_unique_ptr make_arrow_schema(
        std::string_view format,
        std::string_view name,
        std::optional<std::span<char>> metadata,
        std::optional<ArrowFlag> flags,
        C children,
        D dictionary
    )
    {
        SPARROW_ASSERT_FALSE(format.empty())
        SPARROW_ASSERT_TRUE(std::ranges::none_of(
            children,
            [](const auto& child)
            {
                return child == nullptr;
            }
        ))

        arrow_schema_unique_ptr schema = default_arrow_schema();

        schema->flags = flags.has_value() ? static_cast<int64_t>(flags.value()) : 0;
        schema->n_children = static_cast<int64_t>(children.size());

        schema->private_data = new arrow_schema_private_data<Allocator>(
            format,
            name,
            metadata,
            std::move(children),
            std::move(dictionary)
        );

        const auto private_data = static_cast<arrow_schema_private_data<Allocator>*>(schema->private_data);
        schema->format = private_data->format();
        schema->name = private_data->name();
        schema->metadata = private_data->metadata();
        schema->children = private_data->children();
        schema->dictionary = private_data->dictionary();
        schema->release = delete_schema<Allocator>;
        return schema;
    };
}
