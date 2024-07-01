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

#include <cstdint>

#ifndef ARROW_C_DATA_INTERFACE
#    define ARROW_C_DATA_INTERFACE

extern "C"
{
    struct ArrowSchema
    {
        // Array type description
        const char* format;
        const char* name;
        const char* metadata;
        int64_t flags;
        int64_t n_children;
        struct ArrowSchema** children;
        struct ArrowSchema* dictionary;

        // Release callback
        void (*release)(struct ArrowSchema*);
        // Opaque producer-specific data
        void* private_data;
    };

    struct ArrowArray
    {
        // Array data description
        int64_t length;
        int64_t null_count;
        int64_t offset;
        int64_t n_buffers;
        int64_t n_children;
        const void** buffers;
        struct ArrowArray** children;
        struct ArrowArray* dictionary;

        // Release callback
        void (*release)(struct ArrowArray*);
        // Opaque producer-specific data
        void* private_data;
    };

}  // extern "C"

#endif  // ARROW_C_DATA_INTERFACE

namespace sparrow
{
    enum class ArrowFlag : int64_t
    {
        DICTIONARY_ORDERED = 1,  // For dictionary-encoded types, whether the ordering of dictionary indices
                                 // is semantically meaningful.
        NULLABLE = 2,            // Whether this field is semantically nullable (regardless of whether it
                                 // actually has null values).
        MAP_KEYS_SORTED = 4      // For map types, whether the keys within each map value are sorted.
    };
}  // namespace sparrow
