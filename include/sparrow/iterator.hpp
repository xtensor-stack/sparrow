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

#include <compare>
#include <iterator>
#include <memory>
#include <type_traits>

namespace sparrow
{

    namespace detail
    {
        // Since C++20, iterators do not require dereference to return an
        // lvalue, and therefore they do not require to provide operator->.
        // However, because it is such a common operation, and because we
        // don't want to propagate special cases for lvalue / rvalue handling
        // everywhere in generic code, the iterator_base class provides an
        // implementation of operator->, that returns either the address of
        // an lvalue, or the address of a proxy on an rvalue.

        // operator-> dispatcher for proxy references
        template <class Reference, class Pointer>
        struct operator_arrow_dispatcher
        {
            struct proxy
            {
                explicit proxy(const Reference& x)
                    : m_reference(x)
                {
                }

                explicit proxy(Reference&& x)
                    : m_reference(std::move(x))
                {
                }

                Reference* operator->()
                {
                    return std::addressof(m_reference);
                }

            private:

                Reference m_reference;
            };

            using result_type = proxy;

            static result_type apply(const Reference& x)
            {
                return result_type(x);
            }

            static result_type apply(Reference&& x)
            {
                return result_type(std::move(x));
            }
        };

        // operator-> dispatcher for true references
        template <class T, class Pointer>
        struct operator_arrow_dispatcher<T&, Pointer>
        {
            using result_type = Pointer;

            static result_type apply(T& x)
            {
                return std::addressof(x);
            }
        };
    }

    // Base class implementing the interface of a standard iterator
    // that models the given IteratorConcept. Only the specializations
    // for specific IteratorConcept values are defined. Currently,
    // output_iterator_tag and input_iterator_tag are not supported.
    // This allows a lot of simplifications regarding the definitions of
    // postfix increment operators and related methods.
    template
    <
        class Derived,
        class IteratorConcept,
        class Element,
        class Reference,
        class Difference
    >
    class iterator_root_base;

    // Utility class for accessing private methods
    // of iterators.
    class iterator_access
    {
        template <class It>
        static typename It::reference dereference(const It& it)
        {
            return it.dereference();
        }

        template <class It>
        static void increment(It& it)
        {
            it.increment();
        }

        template <class It>
        static void decrement(It& it)
        {
            it.decrement();
        }

        template <class It>
        static void advance(It& it, typename It::difference_type n)
        {
            it.advance(n);
        }

#if _WIN32 // Workaround for access issue with msvc, when friend is involved
    public:
#endif
        template <class It>
        static typename It::difference_type
        distance_to(const It& lhs, const It& rhs)
        {
            return lhs.distance_to(rhs);
        }

        template <class It>
        static bool equal(const It& lhs, const It& rhs)
        {
            return lhs.equal(rhs);
        }

        template <class It>
        static bool less_than(const It& lhs, const It& rhs)
        {
            return lhs.less_than(rhs);
        }
#if _WIN32 // Workaround for access issue with msvc, when friend is involved
    private:
#endif
        template <class Derived, class E, class T, class R, class D>
        friend class iterator_root_base;

        template <class It>
        friend typename It::difference_type operator-(const It&, const It&);

        template <class It>
        friend bool operator==(const It&, const It&);

        template <class It>
        friend std::strong_ordering operator<=>(const It&, const It&);
    };

    // Specialization of iterator_root_base for forward iterator.
    template
    <
        class Derived,
        class Element,
        class Reference,
        class Difference
    >
    class iterator_root_base<Derived, Element, std::forward_iterator_tag, Reference, Difference>
    {
    private:

        using operator_arrow_dispatcher = detail::operator_arrow_dispatcher<Reference, Element*>;

    public:

        using derived_type = Derived;
        using value_type = std::remove_cv_t<Element>;
        using reference = Reference;
        using pointer = typename operator_arrow_dispatcher::result_type;
        using difference_type = Difference;
        using iterator_category = std::forward_iterator_tag;
        using iterator_concept = std::forward_iterator_tag;

        reference operator*() const
        {
            return iterator_access::dereference(this->derived());
        }

        pointer operator->() const
        {
            return operator_arrow_dispatcher::apply(*this->derived());
        }

        derived_type& operator++()
        {
            iterator_access::increment(this->derived());
            return this->derived();
        }

        derived_type operator++(int)
        {
            derived_type tmp(this->derived());
            ++*this;
            return tmp;
        }

        friend inline bool operator==(const derived_type& lhs, const derived_type& rhs)
        {
            return iterator_access::equal(lhs, rhs);
        }

    protected:

        derived_type& derived()
        {
            return *static_cast<derived_type*>(this);
        }

        const derived_type& derived() const
        {
            return *static_cast<const derived_type*>(this);
        }
    };

    // Specialization of iterator_root_base for bidirectional iterator.
    template
    <
        class Derived,
        class Element,
        class Reference,
        class Difference
    >
    class iterator_root_base<Derived, Element, std::bidirectional_iterator_tag, Reference, Difference>
        : public iterator_root_base<Derived, Element, std::forward_iterator_tag, Reference, Difference>
    {
    public:

        using base_type = iterator_root_base<Derived, Element, std::forward_iterator_tag, Reference, Difference>;
        using derived_type = typename base_type::derived_type;
        using iterator_category = std::bidirectional_iterator_tag;
        using iterator_concept = std::bidirectional_iterator_tag;

        derived_type& operator--()
        {
            iterator_access::decrement(this->derived());
            return this->derived();
        }

        derived_type operator--(int)
        {
            derived_type tmp(this->derived());
            --*this;
            return tmp;
        }
    };

    // Specialization of iterator_root_base for random access iterator.
    template
    <
        class Derived,
        class Element,
        class Reference,
        class Difference
    >
    class iterator_root_base<Derived, Element, std::random_access_iterator_tag, Reference, Difference>
        : public iterator_root_base<Derived, Element, std::bidirectional_iterator_tag, Reference, Difference>
    {
    public:

        using base_type = iterator_root_base<Derived, Element, std::bidirectional_iterator_tag, Reference, Difference>;
        using derived_type = typename base_type::derived_type;
        using reference = typename base_type::reference;
        using difference_type = typename base_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;
        using iterator_concept = std::random_access_iterator_tag;

        derived_type& operator+=(difference_type n)
        {
            iterator_access::advance(this->derived(), n);
            return this->derived();
        }

        friend inline derived_type operator+(const derived_type& it, difference_type n)
        {
            derived_type tmp(it);
            return tmp += n;
        }

        friend inline derived_type operator+(difference_type n, const derived_type& it)
        {
            derived_type tmp(it);
            return tmp += n;
        }

        derived_type& operator-=(difference_type n)
        {
            iterator_access::advance(this->derived(), -n);
            return this->derived();
        }

        friend inline derived_type operator-(const derived_type& it, difference_type n)
        {
            derived_type tmp(it);
            return tmp -= n;
        }

        reference operator[](difference_type n) const
        {
            return iterator_access::dereference(this->derived() + n);
        }

        friend inline difference_type operator-(const derived_type& lhs, const derived_type& rhs)
        {
            return iterator_access::distance_to(rhs, lhs);
        }

        friend inline std::strong_ordering operator<=>(const derived_type& lhs, const derived_type& rhs)
        {
            if (iterator_access::less_than(lhs, rhs)) { return std::strong_ordering::less; }
            else if (iterator_access::equal(lhs, rhs)) { return std::strong_ordering::equal; }
            else { return std::strong_ordering::greater; }
        }
    };

    // Specialization of iterator_root_base for contiguous iterator.
    template
    <
        class Derived,
        class Element,
        class Reference,
        class Difference
    >
    class iterator_root_base<Derived, Element, std::contiguous_iterator_tag, Reference, Difference>
        : public iterator_root_base<Derived, Element, std::random_access_iterator_tag, Reference, Difference>
    {
    public:

#ifdef __APPLE__
        // Apple Clang wrong implementation of contiguous_iterator concept
        using base_type = iterator_root_base<Derived, Element, std::random_access_iterator_tag, Reference, Difference>;
        using element_type = typename base_type::value_type;
#endif
        using iterator_concept = std::contiguous_iterator_tag;
    };


    /*
     * @class iterator_base
     * @brief Base class for iterator
     *
     * iterator_base is a CRTP base class that implements the interface of standard iterators,
     * based on functions that must be supplied by a derived iterator class.
     *
     * The derived iterator class must define methods implementing the iterator's core behavior.
     * The following table describes the expressions that must be valid, depending on the concept
     * modeled by the derived iterator.
     *
     *
     *  |    Concept    |     Expression      |        Implementation        |
     *  | ------------- | ------------------- | ---------------------------- |
     *  | forward       | it.dereference()    | Access the value referred to |
     *  | forward       | it.equal(jit)       | Compare for equality         |
     *  | forward       | it.increment()      | Advance by one position      |
     *  | bidirectional | it.decrement()      | Retreat by one position      |
     *  | random access | it.advance(n)       | Advance by n positions       |
     *  | random access | it.distance_to(jit) | Compute the distance         |
     *  | random access | it.less_than(jit)   | Compare for inequality       |
     *
     * These methods should be private and the derived iterator should declare sparrow::iterator_access
     * as a friend class so that `iterator_base` implementation can access those methods.
     *
     * @tparam Derived The type inheriting from this CRTP base class
     * @tparam Element The element type of the iterator, used to compute the `value_type` inner type.
     * Element should be `value_type` iterators, and `const value_type` for const iterators.
     * @tparam IteratorConcept The tag corresponding to the concept satisfied by the iterator.
     * `input_iterator_tag` and `output_iterator_tag` are not supported.
     * @tparam Reference The return type of the dereferencing operation. By default, `Element&`.
     * @tparam Difference The unsigned integer used as the `difference_type`. By default, `std::ptrdiff_t`.
     *
     * @remark By default, the `iterator_category` inner type is the same as the `iterator_concept` inner type
     * (except for contiguous_iterator, where it is `random_access_iterator_tag`). This can be overloaded
     * in the inheriting iterator class.
     */
    template
    <
        class Derived,
        class Element,
        class IteratorConcept,
        class Reference = Element&,
        class Difference = std::ptrdiff_t
    >
    class iterator_base
        : public iterator_root_base<Derived, Element, IteratorConcept, Reference, Difference>
    {
    };
}
