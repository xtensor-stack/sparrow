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

#include <concepts>
#include <list>
#include <variant>
#include <vector>

#include <sparrow/mp_utils.hpp>

namespace sparrow
{

    /////////////////////////////////////////////////////////////////////////////
    // Type predicates


    static_assert(mpl::type_wrapper<std::type_identity_t, int>);
    static_assert(mpl::type_wrapper<mpl::typelist, int>);

    static_assert(mpl::ct_type_predicate<std::is_integral, float>);
    static_assert(mpl::callable_type_predicate<mpl::predicate::same_as<int>, float>);

    static constexpr mpl::ct_type_predicate_to_callable<std::is_integral> object_tpredicate;
    static_assert(object_tpredicate(mpl::typelist<int>{}));


    static_assert(mpl::callable_type_predicate<mpl::ct_type_predicate_to_callable<std::is_integral>, int>);
    static constexpr auto is_integral = mpl::as_predicate<std::is_integral>();
    static_assert(mpl::callable_type_predicate<decltype(is_integral), int>);


    static constexpr auto some_types = mpl::typelist<int, float>{};
    static constexpr auto same_as_int = mpl::predicate::same_as<int>{};
    static_assert(same_as_int(mpl::typelist<int>{}));
    static_assert(mpl::any_of(some_types, same_as_int) == true);
    static_assert(mpl::all_of(some_types, same_as_int) == false);

    //////////////////////////////////////////////////////////////////////////////
    // Type-list

    using test_list = mpl::typelist<int, char>;

    struct not_a_list
    {
    };

    static_assert(mpl::any_typelist<test_list>);
    static_assert(not mpl::any_typelist<not_a_list>);
    static_assert(mpl::size(test_list{}) == 2);

    using test_list_extended = mpl::append_t<test_list, float, double>;
    static_assert(mpl::size(test_list_extended{}) == 4);
    static_assert(std::same_as<test_list_extended, mpl::typelist<int, char, float, double>>);

    using test_list_2 = mpl::typelist<float, double>;
    using test_list_extended_2 = mpl::append_t<test_list, test_list_2>;
    static_assert(mpl::size(test_list_extended_2{}) == 4);
    static_assert(std::same_as<test_list_extended_2, mpl::typelist<int, char, float, double>>);

    static_assert(std::same_as<mpl::rename<test_list, std::variant>, std::variant<int, char>>);

    //////////////////////////////////////////////////////////////////////////////
    // Algorithm

    // any_of
    static_assert(mpl::any_of(test_list{}, mpl::predicate::same_as<int>{}));
    static_assert(mpl::any_of(test_list{}, mpl::predicate::same_as<char>{}));
    static_assert(not mpl::any_of(test_list{}, mpl::predicate::same_as<float>{}));
    static_assert(not mpl::any_of(test_list{}, mpl::predicate::same_as<std::vector<int>>{}));
    static_assert(mpl::any_of(test_list{}, mpl::as_predicate<std::is_integral>()));
    static_assert(mpl::any_of<std::is_integral>(test_list{}));
    static_assert(not mpl::any_of<std::is_floating_point>(test_list{}));

    // all_of
    static_assert(mpl::all_of(mpl::typelist<int, int, int, int, int>{}, mpl::predicate::same_as<int>{}));
    static_assert(
        not mpl::all_of(mpl::typelist<float, int, float, int, float>{}, mpl::predicate::same_as<float>{})
    );
    static_assert(mpl::all_of(mpl::typelist<>{}, mpl::predicate::same_as<int>{}));
    static_assert(mpl::all_of(test_list{}, mpl::as_predicate<std::is_integral>()));
    static_assert(mpl::all_of<std::is_integral>(test_list{}));
    static_assert(not mpl::all_of<std::is_floating_point>(test_list{}));

    // find_if
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<int>{}) == 0);
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<char>{}) == 1);
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<float>{}) == size(test_list{}));
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<std::vector<int>>{}) == size(test_list{}));
    static_assert(mpl::find_if<std::is_integral>(test_list{}) == 0);
    static_assert(mpl::find_if(test_list{}, mpl::as_predicate<std::is_integral>()) == 0);
    static_assert(mpl::find_if<std::is_floating_point>(test_list{}) == size(test_list{}));
    static_assert(mpl::find_if(test_list{}, mpl::as_predicate<std::is_floating_point>()) == size(test_list{}));

    // find
    static_assert(mpl::find<int>(test_list{}) == 0);
    static_assert(mpl::find<char>(test_list{}) == 1);
    static_assert(mpl::find<float>(test_list{}) == size(test_list{}));
    static_assert(mpl::find<std::vector<int>>(test_list{}) == size(test_list{}));

    // contains
    static_assert(mpl::contains<int>(test_list{}));
    static_assert(mpl::contains<char>(test_list{}));
    static_assert(not mpl::contains<float>(test_list{}));
    static_assert(not mpl::contains<double>(test_list{}));

    // transfrom
    static_assert(std::same_as<mpl::typelist<int*, char*>, mpl::transform<std::add_pointer_t, test_list>>);

    // is_unique_ptr
    static_assert(mpl::is_unique_ptr<std::unique_ptr<int>>);
    static_assert(not mpl::is_unique_ptr<int>);
    static_assert(not mpl::is_unique_ptr<std::shared_ptr<int>>);
    static_assert(not mpl::is_unique_ptr<std::weak_ptr<int>>);
    static_assert(not mpl::is_unique_ptr<int*>);

    // is_shared_ptr
    static_assert(mpl::is_shared_ptr<std::shared_ptr<int>>);
    static_assert(not mpl::is_shared_ptr<int>);
    static_assert(not mpl::is_shared_ptr<std::unique_ptr<int>>);
    static_assert(not mpl::is_shared_ptr<std::weak_ptr<int>>);
    static_assert(not mpl::is_shared_ptr<int*>);

    // is_smart_ptr
    static_assert(mpl::is_smart_ptr<std::unique_ptr<int>>);
    static_assert(mpl::is_smart_ptr<std::shared_ptr<int>>);
    static_assert(not mpl::is_smart_ptr<int>);
    static_assert(not mpl::is_smart_ptr<std::weak_ptr<int>>);
    static_assert(not mpl::is_smart_ptr<int*>);
}
