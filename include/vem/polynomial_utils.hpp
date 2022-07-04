#pragma once
#include <array>
#include <cstddef>
namespace vem::polynomials {

namespace one {

// the number of monomials of the given degree
inline size_t num_monomials(size_t degree) { return 1; }

// get the degree of the monomial held at this index
inline size_t monomial_index_degree(size_t index) { return index; }

// the number of monomials up to the specified degree
inline size_t num_monomials_upto(size_t degree) { return degree + 1; }
// get the index for the monomial x^a y^b
inline size_t exponents_to_index(size_t a) { return a; }
// returns [a,b] to represent the monomial x^a y^b that is represented by index
inline size_t index_to_exponents(size_t index) { return index; }
}  // namespace one
namespace two {

// the number of monomials of the given degree
size_t num_monomials(size_t degree);

// get the degree of the monomial held at this index
size_t monomial_index_degree(size_t index);

// the number of monomials up to the specified degree
size_t num_monomials_upto(size_t degree);
// get the index for the monomial x^a y^b
size_t exponents_to_index(size_t a, size_t b);
// returns [a,b] to represent the monomial x^a y^b that is represented by index
std::array<size_t, 2> index_to_exponents(size_t index);
}  // namespace two
namespace three {

// the number of monomials of the given degree
size_t num_monomials(size_t degree);

// get the degree of the monomial held at this index
size_t monomial_index_degree(size_t index);

// the number of monomials up to the specified degree
size_t num_monomials_upto(size_t degree);
// get the index for the monomial x^a y^b
size_t exponents_to_index(size_t a, size_t b, size_t c);
// returns [a,b] to represent the monomial x^a y^b that is represented by index
std::array<size_t, 3> index_to_exponents(size_t index);
}  // namespace three

template <int D>
// the number of monomials of the given degree
size_t num_monomials(size_t degree) {
    if constexpr (D == 2) {
        using namespace two;
        return num_monomials(degree);
    } else if constexpr (D == 3) {
        using namespace three;
        return num_monomials(degree);
    }
}

// get the degree of the monomial held at this index
template <int D>
size_t monomial_index_degree(size_t index) {
    if constexpr (D == 2) {
        using namespace two;
        return monomial_index_degree(index);
    } else if constexpr (D == 3) {
        using namespace three;
        return monomial_index_degree(index);
    }
}

// the number of monomials up to the specified degree
template <int D>
size_t num_monomials_upto(size_t degree) {
    if constexpr (D == 2) {
        using namespace two;
        return num_monomials_upto(degree);
    } else if constexpr (D == 3) {
        using namespace three;
        return num_monomials_upto(degree);
    }
}
// get the index for the monomial x^a y^b
template <int D>
size_t exponents_to_index(const std::array<size_t, D>& v) {
    if constexpr (D == 2) {
        using namespace two;
        return exponents_to_index(v[0], v[1]);
    } else if constexpr (D == 3) {
        using namespace three;
        return exponents_to_index(v[0], v[1], v[2]);
    }
}
template <int D>
// returns [a,b] to represent the monomial x^a y^b that is represented by index
std::array<size_t, D> index_to_exponents(size_t index) {
    if constexpr (D == 2) {
        using namespace two;
        return index_to_exponents(index);
    } else if constexpr (D == 3) {
        using namespace three;
        return index_to_exponents(index);
    }
}
}  // namespace vem::polynomials
