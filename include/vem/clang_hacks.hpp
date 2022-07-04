#pragma once
#include <array>
#include <tuple>

namespace vem {
// names in structured bindings are technically temporaries, and temporaries
// can't be captured in a lambda. because clang conforms too much to this asepct
// of the standard we can't do auto [a,b,c] = func(); auto func = [a,b,c](...)
// {...}; instead we'll be doing int a,b,c; assign_array_to_tuple(func(),
// std::tie(a,b,c))
namespace internal {
template <typename... TTypes, std::size_t D, typename AType, int... N>
void assign_array_to_tuple(std::integer_sequence<int, N...>,
                           const std::array<AType, D>& arr,
                           std::tuple<TTypes&...>&& tup) {
    auto eq = [](auto& a, const auto& b) { a = b; };
    (eq(std::get<N>(tup), arr[N]), ...);
}
template <typename... TTypes, typename... DTypes, int... N>
void assign_array_to_tuple(std::integer_sequence<int, N...>,
                           const std::tuple<DTypes...>& arr,
                           std::tuple<TTypes&...>&& tup) {
    auto eq = [](auto& a, const auto& b) { a = b; };
    (eq(std::get<N>(tup), std::get<N>(arr)), ...);
}
}  // namespace internal

template <typename... TTypes, std::size_t D, typename AType>
void assign_array_to_tuple(const std::array<AType, D>& arr,
                           std::tuple<TTypes&...>&& tup) {
    static_assert(sizeof...(TTypes) == D);
    internal::assign_array_to_tuple(
        std::make_integer_sequence<int, int(D)>{}, arr,
        std::forward<std::tuple<TTypes&...>&&>(tup));
}

template <typename... TTypes, typename... DTypes>
void assign_array_to_tuple(const std::tuple<DTypes...>& arr,
                           std::tuple<TTypes&...>&& tup) {
    constexpr static int D = sizeof...(DTypes);
    static_assert(sizeof...(TTypes) == D);
    internal::assign_array_to_tuple(
        std::make_integer_sequence<int, D>{}, arr,
        std::forward<std::tuple<TTypes&...>&&>(tup));
}
}  // namespace vem
