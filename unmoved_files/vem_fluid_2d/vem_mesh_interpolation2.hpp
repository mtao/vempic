#pragma once
#include <mandoline/mesh2.hpp>

#include "vem_mesh2.hpp"

struct VEMMesh2FieldBase {
    const mandoline::CutCellMesh<2>& ccm;
    const VEMMesh2& vem;
    template <typename Derived>
    int get_cell(const Eigen::MatrixBase<Derived>& p) const {
        return ccm.cell_index(p);
    }

    template <typename CDerived, typename Derived>
    double eval(const Eigen::MatrixBase<CDerived>& coeffs, int cell,
                const Eigen::MatrixBase<Derived>& p) const {
        size_t off = vem.coefficient_size();
        if (cell > vem.num_cells() || cell < 0) {
            // spdlog::warn("Invalid field evaluation at cell {} from point at
            // ",
            //             cell, p.x(), p.y());
            return 0;
        }
        return vem.polynomial_eval<1>(
            cell, p, coeffs.segment(cell * off, off).eval())(0);
    }
    template <typename CDerived, typename Derived>
    double eval(const Eigen::MatrixBase<CDerived>& coeffs,
                const Eigen::MatrixBase<Derived>& p) const {
        int cell = get_cell(p);
        return eval(coeffs, cell, p);
    }
};

struct VEMMesh2ScalarField : public VEMMesh2FieldBase {
    mtao::VecXd coefficients;
    VEMMesh2ScalarField(const VEMMesh2FieldBase& o) : VEMMesh2FieldBase{o} {}

    VEMMesh2ScalarField(const mandoline::CutCellMesh<2>& ccm,
                        const VEMMesh2& vem, const mtao::VecXd& c)
        : VEMMesh2FieldBase{ccm, vem}, coefficients(c) {}
    VEMMesh2ScalarField(const mandoline::CutCellMesh<2>& ccm,
                        const VEMMesh2& vem, int size)
        : VEMMesh2ScalarField(ccm, vem, mtao::VecXd::Zero(size)) {}
    VEMMesh2ScalarField(const mandoline::CutCellMesh<2>& ccm,
                        const VEMMesh2& vem)
        : VEMMesh2ScalarField(ccm, vem, vem.polynomial_size()) {}
    template <typename Derived>
    double operator()(const Eigen::MatrixBase<Derived>& p) const {
        return eval(coefficients, p);
    }
};
struct VEMMesh2VectorField : public VEMMesh2FieldBase {
    mtao::RowVecs2d coefficients;
    VEMMesh2VectorField(const VEMMesh2FieldBase& o) : VEMMesh2FieldBase{o} {}
    VEMMesh2VectorField(const mandoline::CutCellMesh<2>& ccm,
                        const VEMMesh2& vem, const mtao::RowVecs2d& coeffs)
        : VEMMesh2FieldBase{ccm, vem}, coefficients(coeffs) {}
    VEMMesh2VectorField(const mandoline::CutCellMesh<2>& ccm,
                        const VEMMesh2& vem, int size)
        : VEMMesh2VectorField(ccm, vem, mtao::RowVecs2d::Zero(size, 2)) {}
    VEMMesh2VectorField(const mandoline::CutCellMesh<2>& ccm,
                        const VEMMesh2& vem)
        : VEMMesh2VectorField(ccm, vem, vem.polynomial_size()) {}

    template <typename Derived>
    mtao::Vec2d integrate(const Eigen::MatrixBase<Derived>& x,
                          double dt) const {
        // rk2
        mtao::Vec2d p2 = x + .5 * dt * (*this)(x);
        return x + dt * (*this)(p2);
    }
    template <typename Derived>
    mtao::Vec2d operator()(const Eigen::MatrixBase<Derived>& p,
                           int cell) const {
        return {eval(coefficients.col(0), cell, p),
                eval(coefficients.col(1), cell, p)};
    }
    template <typename Derived>
    mtao::Vec2d operator()(const Eigen::MatrixBase<Derived>& p) const {
        int cell = get_cell(p);
        return (*this)(p, cell);
    }
    template <typename Derived>
    void operator()(const Eigen::MatrixBase<Derived>& x, mtao::Vec2d& dxdt) {
        dxdt = (*this)(x);
    }
    void operator()(const std::array<double, 2>& x,
                    std::array<double, 2>& dxdt) {
        mtao::eigen::stl2eigen(dxdt) = (*this)(mtao::eigen::stl2eigen(x));
    }
    // template <typename Derived>
    // mtao::Vec2d operator()(const Eigen::MatrixBase<Derived>& x) {
    //    mtao::Vec2d ret;
    //    this->operator()(x, ret);
    //    return ret;
    //}
    void operator()(const std::array<double, 2>& x, std::array<double, 2>& dxdt,
                    double t) {
        this->operator()(x, dxdt);
    }
};
