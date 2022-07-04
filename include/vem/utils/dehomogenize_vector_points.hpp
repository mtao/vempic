#include <mtao/types.hpp>

namespace vem::utils {
template <int D>
mtao::ColVectors<double, D - 1> dehomogenize_vector_points(
    const mtao::ColVectors<double, D>& P) {
    auto Dat = P.template topRows<D - 1>();
    auto W = P.row(D - 1).transpose();
    mtao::VecXd w = (W.array().abs() > 1e-10).select(1.0 / W.array(), 0.0);
    return Dat * w.asDiagonal();
}
}  // namespace vem::utils
