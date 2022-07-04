
#include <spdlog/spdlog.h>

#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/triangle_monomial_integrals.hpp>

#include "vem/monomial_face_integrals.hpp"
#include "vem/polynomial_utils.hpp"

namespace vem {
namespace internal {

template <typename Derived>
mtao::VecXd monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                    int max_degree,
                                    const Eigen::MatrixBase<Derived> &center,
                                    double scale = 1) {
    const auto &V = mesh.V;
    size_t size = polynomials::three::num_monomials_upto(max_degree);
    Eigen::VectorXd monomial_integrals(size);
    // spdlog::info("Making monomial integrals of size: {} from max degree {}",
    //             monomial_integrals.size(), max_degree);
    monomial_integrals.setZero();
    const auto &F = mesh.triangulated_faces.at(index);
    // spdlog::error("F: {}:{}", fidx, sign);
    // std::cout << F << std::endl;
    for (int j = 0; j < F.cols(); ++j) {
        auto f = F.col(j);

        mtao::Vec3d b = (V.col(f(0)) - center) / scale;
        mtao::Vec3d c = (V.col(f(1)) - center) / scale;
        mtao::Vec3d d = (V.col(f(2)) - center) / scale;

        // std::cout << V.col(f(0)).transpose() << " ||| "
        //          << V.col(f(1)).transpose() << " ||| "
        //          << V.col(f(2)).transpose() << std::endl;
        // std::cout << b.transpose() << " +++++ " << c.transpose() << " +++++ "
        //          << d.transpose() << std::endl;

        auto v = mtao::geometry::triangle_monomial_integrals<double>(max_degree,
                                                                     b, c, d);
        // std::cout << mtao::eigen::stl2eigen(v).transpose() << std::endl;
        // spdlog::info("Monomial integrals made {} values", v.size());
        monomial_integrals += mtao::eigen::stl2eigen(v);
    }
    return monomial_integrals * (scale * scale);
}

template <typename Derived>
mtao::VecXd face_monomial_face_integrals(
    const VEMMesh3 &mesh, int index, int max_degree,
    const Eigen::MatrixBase<Derived> &center, double scale = 1) {
    const auto &V = mesh.V;
    size_t size = polynomials::two::num_monomials_upto(max_degree);
    Eigen::VectorXd monomial_integrals(size);
    // spdlog::info("Making monomial integrals of size: {} from max degree {}",
    //             monomial_integrals.size(), max_degree);
    monomial_integrals.setZero();
    const auto &F = mesh.triangulated_faces.at(index);
    // spdlog::error("F: {}:{}", fidx, sign);
    // std::cout << F << std::endl;
    const auto &UV = mesh.face_frames.at(index);
    mtao::Vec2d C = UV.transpose() * center;
    for (int j = 0; j < F.cols(); ++j) {
        auto f = F.col(j);

        mtao::Vec2d a = (UV.transpose() * V.col(f(0)) - C) / scale;
        mtao::Vec2d b = (UV.transpose() * V.col(f(1)) - C) / scale;
        mtao::Vec2d c = (UV.transpose() * V.col(f(2)) - C) / scale;
        // std::cout << a.transpose() << " ||| " << b.transpose() << " ||| "
        //          << c.transpose() << std::endl;

        auto v = mtao::geometry::triangle_monomial_integrals<double>(max_degree,
                                                                     a, b, c);
        // spdlog::info("Monomial integrals made {} values", v.size());
        monomial_integrals += mtao::eigen::stl2eigen(v);
    }
    // fix potential orientation issues, integral of the constant function should always be positive here
    if(monomial_integrals(0) < 0) {
        monomial_integrals *= -1;
    }
    return monomial_integrals * (scale * scale);
}
}  // namespace internal

mtao::VecXd monomial_face_integrals(const VEMMesh3 &mesh, int face_index,
                                    int index, int max_degree) {
    if (index >= mesh.face_count()) {
        spdlog::warn(
            "monomial face integrals called on invalid face index {} of {}",
            index, mesh.face_count());
        return {};
    }
    return internal::monomial_face_integrals(mesh, index, max_degree,
                                             mesh.C.col(face_index));
}
mtao::VecXd monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                    int max_degree, const mtao::Vec3d &center) {
    return internal::monomial_face_integrals(mesh, index, max_degree, center);
}

mtao::VecXd scaled_monomial_face_integrals(const VEMMesh3 &mesh, int face_index,
                                           int index, double scale,
                                           int max_degree) {
    if (index >= mesh.face_count()) {
        spdlog::warn(
            "monomial face integrals called on invalid face index {} of {}",
            index, mesh.face_count());
        return {};
    }
    return internal::monomial_face_integrals(mesh, index, max_degree,
                                             mesh.C.col(face_index), scale);
}
mtao::VecXd scaled_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec3d &center) {
    return internal::monomial_face_integrals(mesh, index, max_degree, center,
                                             scale);
}

mtao::VecXd face_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                         int max_degree) {
    if (index >= mesh.face_count()) {
        spdlog::warn(
            "face_monomial face integrals called on invalid face index {} of "
            "{}",
            index, mesh.face_count());
        return {};
    }
    return internal::face_monomial_face_integrals(mesh, index, max_degree,
                                                  mesh.FC.col(index));
}
mtao::VecXd face_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                         int max_degree,
                                         const mtao::Vec3d &center) {
    return internal::face_monomial_face_integrals(mesh, index, max_degree,
                                                  center);
}

mtao::VecXd scaled_face_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                                double scale, int max_degree) {
    if (index >= mesh.face_count()) {
        spdlog::warn(
            "face_monomial face integrals called on invalid face index {} of "
            "{}",
            index, mesh.face_count());
        return {};
    }
    return internal::face_monomial_face_integrals(mesh, index, max_degree,
                                                  mesh.FC.col(index), scale);
}
mtao::VecXd scaled_face_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                                double scale, int max_degree,
                                                const mtao::Vec3d &center) {
    return internal::face_monomial_face_integrals(mesh, index, max_degree,
                                                  center, scale);
}

}  // namespace vem
