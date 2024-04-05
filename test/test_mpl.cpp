#include "doctest/doctest.h"

#include <sparrow/mp_utils.hpp>

#include <vector>

namespace sparrow
{
    static_assert(mpl::ct_type_predicate<std::is_integral, float>);
    static_assert(mpl::callable_type_predicate< mpl::predicate::same_as<int>, float >);



    using test_list = mpl::typelist< int, char >;
    struct not_a_list { };
    static_assert(mpl::any_typelist<test_list>);
    static_assert(not mpl::any_typelist<not_a_list>);
    static_assert(mpl::size(test_list{}) == 2);


    static_assert(mpl::any_of(test_list{}, mpl::predicate::same_as<int>{}));
    static_assert(mpl::any_of(test_list{}, mpl::predicate::same_as<char>{}));
    static_assert(not mpl::any_of(test_list{}, mpl::predicate::same_as<float>{}));
    static_assert(not mpl::any_of(test_list{}, mpl::predicate::same_as<std::vector<int>>{}));
    static_assert(mpl::any_of(test_list{}, mpl::as_predicate<std::is_integral>()));
    static_assert(mpl::any_of<std::is_integral>(test_list{}));
    static_assert(not mpl::any_of<std::is_floating_point>(test_list{}));

    static_assert(mpl::all_of(mpl::typelist<int, int, int, int, int>{}, mpl::predicate::same_as<int>{}));
    static_assert(not mpl::all_of(mpl::typelist<float, int, float, int, float>{}, mpl::predicate::same_as<float>{}));
    static_assert(mpl::all_of(mpl::typelist<>{}, mpl::predicate::same_as<int>{}));
    static_assert(mpl::all_of(test_list{}, mpl::as_predicate<std::is_integral>()));
    static_assert(mpl::all_of<std::is_integral>(test_list{}));
    static_assert(not mpl::all_of<std::is_floating_point>(test_list{}));


    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<int>{}) == 0);
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<char>{}) == 1);
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<float>{}) >= size(test_list{}));
    static_assert(mpl::find_if(test_list{}, mpl::predicate::same_as<std::vector<int>>{}) >= size(test_list{}));
    static_assert(mpl::find_if<std::is_integral>(test_list{}) == 0);
    static_assert(mpl::find_if(test_list{}, mpl::as_predicate<std::is_integral>()) == 0);
    static_assert(mpl::find_if<std::is_floating_point>(test_list{}) == size(test_list{}));
    static_assert(mpl::find_if(test_list{}, mpl::as_predicate<std::is_floating_point>()) == size(test_list{}));

    static_assert(mpl::find<int>(test_list{}) == 0);
    static_assert(mpl::find<char>(test_list{}) == 1);
    static_assert(mpl::find<float>(test_list{}) >= size(test_list{}));
    static_assert(mpl::find<std::vector<int>>(test_list{}) >= size(test_list{}));


    static_assert(mpl::contains<int>(test_list{}));
    static_assert(mpl::contains<char>(test_list{}));
    static_assert(not mpl::contains<float>(test_list{}));
    static_assert(not mpl::contains<double>(test_list{}));




}