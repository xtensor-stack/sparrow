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
#include <memory>
#include <memory_resource>
#include <type_traits>
#include <typeindex>
#include <variant>

namespace sparrow
{
    /*
     * allocator concept, based on the requirements specified by the standard:
     * https://en.cppreference.com/w/cpp/named_req/Allocator.
     */
    template <class T>
    concept allocator = std::copy_constructible<std::remove_cvref_t<T>>
                        and std::move_constructible<std::remove_cvref_t<T>>
                        and std::equality_comparable<std::remove_cvref_t<T>>
                        and requires(
                            std::remove_cvref_t<T>& alloc,
                            typename std::remove_cvref_t<T>::value_type* p,
                            std::size_t n
                        ) {
                                typename std::remove_cvref_t<T>::value_type;
                                {
                                    alloc.allocate(n)
                                } -> std::same_as<typename std::remove_cvref_t<T>::value_type*>;
                                { alloc.deallocate(p, n) } -> std::same_as<void>;
                            };

    /*
     * When the allocator A with value_type T satisfies this concept, any_allocator
     * can store it as a value in a small buffer instead of having to type-erased it
     * (Small Buffer Optimization).
     */
    template <class A, class T>
    concept can_any_allocator_sbo = allocator<A>
                                    && (std::same_as<A, std::allocator<T>>
                                        || std::same_as<A, std::pmr::polymorphic_allocator<T>>);

    template <class... Ts>
    struct overloaded : Ts...
    {
        using Ts::operator()...;
    };
    // Although not required in C++20, clang needs it to build the code below
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

    /*
     * Type erasure class for allocators. This allows to use any kind of allocator
     * (standard, polymorphic) without having to expose it as a template parameter.
     *
     * @tparam T value_type of the allocator
     */
    template <class T>
    class any_allocator
    {
    public:

        using value_type = T;

        any_allocator();
        any_allocator(const any_allocator& rhs);
        any_allocator(any_allocator&&) = default;

        any_allocator& operator=(const any_allocator& rhs) = delete;
        any_allocator& operator=(any_allocator&& rhs) = delete;

        template <class A>
        any_allocator(A&& alloc)
            requires(not std::same_as<std::remove_cvref_t<A>, any_allocator> and sparrow::allocator<A>)
            : m_storage(make_storage(std::forward<A>(alloc)))
        {
        }

        [[nodiscard]] T* allocate(std::size_t n);
        void deallocate(T* p, std::size_t n);

        any_allocator select_on_container_copy_construction() const;

        bool equal(const any_allocator& rhs) const;

    private:

        struct interface
        {
            [[nodiscard]] virtual T* allocate(std::size_t) = 0;
            virtual void deallocate(T*, std::size_t) = 0;
            virtual std::unique_ptr<interface> clone() const = 0;
            virtual bool equal(const interface&) const = 0;
            virtual ~interface() = default;
        };

        template <allocator A>
        struct impl : interface
        {
            A m_alloc;

            explicit impl(A alloc)
                : m_alloc(std::move(alloc))
            {
            }

            [[nodiscard]] T* allocate(std::size_t n) override
            {
                return m_alloc.allocate(n);
            }

            void deallocate(T* p, std::size_t n) override
            {
                m_alloc.deallocate(p, n);
            }

            std::unique_ptr<interface> clone() const override
            {
                return std::make_unique<impl<A>>(m_alloc);
            }

            bool equal(const interface& rhs) const override
            {
                if (std::type_index(typeid(*this)) == std::type_index(typeid(rhs)))
                {
                    return m_alloc == static_cast<const impl<A>*>(std::addressof(rhs))->m_alloc;
                }
                return false;
            }
        };

        using storage_type = std::
            variant<std::allocator<T>, std::pmr::polymorphic_allocator<T>, std::unique_ptr<interface>>;

        template <class A>
        std::unique_ptr<interface> make_storage(A&& alloc) const
        {
            return std::make_unique<impl<std::decay_t<A>>>(std::forward<A>(alloc));
        }

        template <class A>
            requires can_any_allocator_sbo<A, T>
        A&& make_storage(A&& alloc) const
        {
            return std::forward<A>(alloc);
        }

        storage_type copy_storage(const storage_type& rhs) const
        {
            return std::visit(
                overloaded{
                    [](const auto& arg) -> storage_type
                    {
                        return {std::decay_t<decltype(arg)>(arg)};
                    },
                    [](const std::unique_ptr<interface>& arg) -> storage_type
                    {
                        return {arg->clone()};
                    }
                },
                rhs
            );
        }

        template <class F>
        decltype(auto) visit_storage(F&& f)
        {
            return std::visit(
                [&f](auto&& arg)
                {
                    using A = std::decay_t<decltype(arg)>;
                    if constexpr (can_any_allocator_sbo<A, T>)
                    {
                        return f(arg);
                    }
                    else
                    {
                        return f(*arg);
                    }
                },
                m_storage
            );
        }

        storage_type m_storage;
    };

    /********************************
     * any_allocator implementation *
     ********************************/

    template <class T>
    any_allocator<T>::any_allocator()
        : m_storage(make_storage(std::allocator<T>()))
    {
    }

    template <class T>
    any_allocator<T>::any_allocator(const any_allocator& rhs)
        : m_storage(copy_storage(rhs.m_storage))
    {
    }

    template <class T>
    [[nodiscard]] T* any_allocator<T>::allocate(std::size_t n)
    {
        return visit_storage(
            [n](auto& allocator)
            {
                return allocator.allocate(n);
            }
        );
    }

    template <class T>
#if defined(_MSC_VER) && !defined(__clang__)  // MSVC
    __declspec(no_sanitize_address)
#else
#    if defined(__has_feature)
#        if __has_feature(address_sanitizer)
    __attribute__((no_sanitize("address")))
#        endif
#    endif
#endif
    void
    any_allocator<T>::deallocate(T* p, std::size_t n)
    {
        return visit_storage(
            [n, p](auto& allocator)
            {
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
                return allocator.deallocate(p, n);
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
            }
        );
    }

    template <class T>
    any_allocator<T> any_allocator<T>::select_on_container_copy_construction() const
    {
        return any_allocator(*this);
    }

    template <class T>
    bool any_allocator<T>::equal(const any_allocator& rhs) const
    {
        // YOLO!!
        return std::visit(
            [&rhs](auto&& arg)
            {
                using A = std::decay_t<decltype(arg)>;
                if constexpr (can_any_allocator_sbo<A, T>)
                {
                    return std::visit(
                        [&arg](auto&& arg2)
                        {
                            using A2 = std::decay_t<decltype(arg2)>;
                            if constexpr (can_any_allocator_sbo<A2, T> && std::same_as<A, A2>)
                            {
                                return arg == arg2;
                            }
                            else
                            {
                                return false;
                            }
                        },
                        rhs.m_storage
                    );
                }
                else
                {
                    return std::visit(
                        [&arg](auto&& arg2)
                        {
                            using A2 = std::decay_t<decltype(arg2)>;
                            if constexpr (can_any_allocator_sbo<A2, T>)
                            {
                                return false;
                            }
                            else
                            {
                                return arg->equal(*arg2);
                            }
                        },
                        rhs.m_storage
                    );
                }
            },
            m_storage
        );
    }

    template <class T>
    bool operator==(const any_allocator<T>& lhs, const any_allocator<T>& rhs)
    {
        return lhs.equal(rhs);
    }
}
