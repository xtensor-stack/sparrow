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

#include <list>

#include "sparrow/array_data_concepts.hpp"

namespace sparrow
{
    static_assert(arrow_layout<fixed_size_layout<int>>);
    static_assert(arrow_layout<variable_size_binary_layout<std::string, const std::string_view>>);
    static_assert(arrow_layout<dictionary_encoded_layout<
                      size_t,
                      variable_size_binary_layout<std::string, const std::string_view>>>);
    static_assert(!arrow_layout<std::string>);

    static_assert(range_of_arrow_base_type_extended<std::vector<int>>);
    static_assert(range_of_arrow_base_type_extended<std::vector<std::string>>);
    static_assert(range_of_arrow_base_type_extended<std::vector<std::string_view>>);
    static_assert(!range_of_arrow_base_type_extended<std::vector<std::list<int>>>);

    static_assert(range_for_array_data<std::vector<int>>);
    static_assert(range_for_array_data<std::vector<std::string>>);
    static_assert(range_for_array_data<std::vector<std::string_view>>);
    static_assert(range_for_array_data<std::vector<std::list<int>>>);
    static_assert(range_for_array_data<std::vector<std::reference_wrapper<std::string>>>);
    static_assert(!range_for_array_data<std::vector<std::vector<std::vector<int>>>>);

    static_assert(constant_range_for_array_data<const std::vector<int>>);
    static_assert(constant_range_for_array_data<const std::vector<std::string>>);
    static_assert(!constant_range_for_array_data<std::vector<int>>);
}
