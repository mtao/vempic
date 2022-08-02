#include <spdlog/spdlog.h>

#include <mtao/eigen/stack.hpp>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/tetrahedron_monomial_integrals.hpp>

#include "vem/three/monomial_cell_integrals.hpp"
#include "vem/polynomials/utils.hpp"

namespace vem::three {
namespace internal {

template <typename Derived>
mtao::VecXd monomial_cell_integrals(const VEMMesh3 &mesh, int index,
                                    int max_degree,
                                    const Eigen::MatrixBase<Derived> &center,
                                    double scale = 1) {
    const auto &V = mesh.V;
    const auto &Fs = mesh.triangulated_faces;
    const auto &cell_boundaries = mesh.cell_boundary_map.at(index);
    size_t size = polynomials::three::num_monomials_upto(max_degree);
    Eigen::VectorXd monomial_integrals(size);
    // spdlog::info("Making monomial integrals of size: {} from max degree {}",
    //             monomial_integrals.size(), max_degree);
    monomial_integrals.setZero();
    //std::cout << "Center: " << center.transpose() << std::endl;
    //std::cout << "DX: " << mesh.dx()<< "Scale:"<< scale << std::endl;
    for (auto &&[fidx, sgn] : cell_boundaries) {
        const auto &F = Fs.at(fidx);
        double sign = sgn ? -1 : 1;
         //spdlog::error("F: {}:{}", fidx, sign);
         //std::cout << F << std::endl;
        for (int j = 0; j < F.cols(); ++j) {
            auto f = F.col(j);

            mtao::Vec3d b = (V.col(f(0)) - center) / scale;
            mtao::Vec3d c = (V.col(f(1)) - center) / scale;
            mtao::Vec3d d = (V.col(f(2)) - center) / scale;

            //std::cout << V.col(f(0)).transpose() << " ||| " << V.col(f(1)).transpose() << " ||| " << V.col(f(2)).transpose() << std::endl;
            //std::cout << b.transpose() << " +++++ " << c.transpose() << " +++++ " << d.transpose() << std::endl;
std:

            auto v = mtao::geometry::tetrahedron_monomial_integrals<double>(
                max_degree, mtao::Vec3d::Zero(), b, c, d);
             //std::cout << sign * mtao::eigen::stl2eigen(v).transpose()
             //         << std::endl;
            monomial_integrals += sign * mtao::eigen::stl2eigen(v);
        }
    }
    return monomial_integrals * (scale * scale * scale);
}
}  // namespace internal

mtao::VecXd monomial_cell_integrals(const VEMMesh3 &mesh, int index,
                                    int max_degree) {
    if (index >= mesh.cell_count()) {
        spdlog::warn(
            "monomial cell integrals called on invalid cell index {} of {}",
            index, mesh.cell_count());
        return {};
    }
    return internal::monomial_cell_integrals(mesh, index, max_degree,
                                             mesh.C.col(index));
}
mtao::VecXd monomial_cell_integrals(const VEMMesh3 &mesh, int index,
                                    int max_degree, const mtao::Vec3d &center) {
    return internal::monomial_cell_integrals(mesh, index, max_degree, center);
}

mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh3 &mesh, int index,
                                           double scale, int max_degree) {
    if (index >= mesh.cell_count()) {
        spdlog::warn(
            "monomial cell integrals called on invalid cell index {} of {}",
            index, mesh.cell_count());
        return {};
    }
    return internal::monomial_cell_integrals(mesh, index, max_degree,
                                             mesh.C.col(index), scale);
}
mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh3 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec3d &center) {
    return internal::monomial_cell_integrals(mesh, index, max_degree, center,
                                             scale);
}
}  // namespace vem
 namespace vem {
mtao::VecXd scaled_monomial_cell_integrals(const three::VEMMesh3 &mesh, int index, double scale, int max_degree)
{
    return three::scaled_monomial_cell_integrals(mesh,index,scale,max_degree);
}
 }
