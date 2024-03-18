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

#include <type_traits>
#include <iterator>

namespace sparrow::mpl
{
    /// A sequence of types, used for meta-programming operations.
    template< class... T >
    struct typelist {};

    /// @returns The count of types contained in a given `typelist`.
    template< class... T >
    constexpr std::size_t size(typelist<T...>) { return sizeof...(T); }

    template< class L, template<class...> class U >
    struct is_type_instance_of : std::false_type {};

    template< template<class...> class L, template<class...> class U, class... T>
        requires std::is_same_v< L<T...>, U<T...> >
    struct is_type_instance_of< L<T...>, U > : std::true_type {};

    /// `true` if `T` is a concrete type template instanciation of `U` which is a type template.
    /// Example: is_type_instant_of_v< std::vector<int>, std::vector > == true
    template< class T, template< class... > class U >
    constexpr bool is_type_instance_of_v = is_type_instance_of<T, U>::value;

    /// Matches any type which is an instance of `typelist`.
    /// Examples:
    ///     stati_assert(any_typelist< typelist<int, float, std::vector<int> > > == true);
    ///     stati_assert(any_typelist< std::vector<int> > == false);
    template< typename TList >
    concept any_typelist = is_type_instance_of_v<TList, typelist>;

    // TODO: document this!
    template <class T, bool is_const>
    struct constify
        : std::conditional<is_const, const T, T>
    {
    };

    template <class T, bool is_const>
    using constify_t = typename constify<T, is_const>::type;

    /// Checks that at least one type in the provided list of is making the provide predicate return `true`.
    /// @returns `true` if for any type T in the type list L, `Predicate{}(typelist<T>) == true`.
    ///          `false` otherwise or if the list is empty.
    template< template<class...> class L, class Predicate, class... T>
        requires any_typelist<L<T...>>
    consteval
    bool any_of(L<T...> list, Predicate predicate)
    {
        return (predicate(typelist<T>{}) || ... || false);
    }

    /// Checks that every type in the provided list of is making the provide predicate return `true`.
    /// @returns `true` if for every type T in the type list L, `Predicate{}(typelist<T>) == true`
    ///          or if the list is empty; `false` otherwise.
    template< template<class...> class L, class Predicate, class... T>
        requires any_typelist<L<T...>>
    consteval
        bool all_of(L<T...> list, Predicate predicate)
    {
        return (predicate(typelist<T>{}) && ... && true);
    }

    /// Compile-time type predicate: `true` if the evaluated type is the same as `T`.
    /// Example:
    ///     static constexpr auto some_types = typelist<int, float>{}
    ///     static constexpr auto same_as_int = same_as<int>{};
    ///     static_assert( any_of(some_types, same_as_int) == true);
    ///     static_assert( all_of(some_types, same_as_int) == false);
    ///
    // TODO: elaborate documentation?
    template<class T>
    struct same_as
    {
        consteval
        bool operator()(mpl::any_typelist auto list) const
        {
            return std::same_as< mpl::typelist<T>, decltype(list) >;
        }

    };

    /// @returns `true` if the provided type list contains `V`
    template< class V, any_typelist L>
    consteval
    bool contains(L list)
    {
        return any_of(list, same_as<V>{});
    }


    /// @returns The index position in the first type in the provided type list `L` that matches the provided predicate,
    ///          or the size of the list if the matching type was not found.
    template< template<class...> class L, class Predicate, class... T>
        requires any_typelist<L<T...>>
    consteval
    std::size_t find_if(L<T...> list, Predicate predicate)
    {
        std::size_t idx = 0;
        auto check = [&](bool match_success){
            if (match_success)
            {
                return true;
            }
            else
            {
                ++idx;
                return false;
            }
        };

        (check(predicate(typelist<T>{})) || ...);

        return idx;
    }

    /// @returns The index position in the type `TypeToFind` in the provided type list `L`,
    ///          or the size of the list if the matching type was not found.
    template< class TypeToFind, any_typelist L>
    consteval
    std::size_t find(L list)
    {
        return find_if(list, same_as<TypeToFind>{});
    }

}
