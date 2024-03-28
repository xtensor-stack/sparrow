#include "doctest/doctest.h"

#include <sparrow/data_traits.hpp>

/////////////////////////////////////////////////////////////////////////////////////////
// Opt-in support for custom C++ representations of arrow data types.

struct MyDataType{};

template<>
struct sparrow::arrow_traits<MyDataType>
{
    static constexpr data_type type_id = sparrow::data_type::INT32;
    using value_type = MyDataType;
    using default_layout = sparrow::fixed_size_layout<MyDataType>;
};


static_assert( sparrow::is_arrow_traits< sparrow::arrow_traits<MyDataType> > );
static_assert( sparrow::any_arrow_type< MyDataType > );

//////////////////////////////////////////////////////////////////////////////////////////
// Base arrow types representations support tests and concept checking.
namespace sparrow
{
    struct {
        template<class T>
        consteval bool operator()(mpl::typelist<T>)
        {
            return is_arrow_base_type<T>;
        }
    } constexpr predicate_is_arrow_base_type;


    static_assert(mpl::all_of(all_base_types_t{}, predicate_is_arrow_base_type));

    struct {
        template<class T>
        consteval bool operator()(mpl::typelist<T>)
        {
            return is_arrow_traits< sparrow::arrow_traits<T> >;
        }
    } constexpr predicate_has_arrow_traits;

    static_assert(mpl::all_of(all_base_types_t{}, predicate_has_arrow_traits));

}
