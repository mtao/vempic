#pragma once
#include "vem/fluidsim_3d/fluidvem3.hpp"

namespace vem::fluidsim_3d {
class OperatorCache {
   public:
    OperatorCache(const FluidVEM3& fvem, bool build = false);
    void build_cache();

    bool built() const { return _built; }

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sample_laplacian()
        const {
        return _sample_laplacian;
    }
    const Eigen::SparseMatrix<double, Eigen::RowMajor>&
    sample_to_poly_codivergence() const {
        return _sample_to_poly_codivergence;
    }
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sample_codivergence()
        const {
        return _sample_codivergence;
    }
    const Eigen::SparseMatrix<double, Eigen::RowMajor>&
    sample_to_poly_dirichlet() const {
        return _sample_to_poly_dirichlet;
    }
    const Eigen::SparseMatrix<double, Eigen::RowMajor>&
    sample_to_poly_gradient() const {
        return _sample_to_poly_gradient;
    }

    const Eigen::SparseMatrix<double, Eigen::RowMajor>& sample_to_poly_l2()
        const {
        return _sample_to_poly_l2;
    }
    const std::vector<std::set<size_t>>& face_coboundary() const {
        return _face_coboundary;
    }

    mtao::VecXd& old_pressure_solution() { return _old_pressure_solution; }

   private:
    const FluidVEM3& _fvem;
    bool _built = false;
    Eigen::SparseMatrix<double, Eigen::RowMajor> _sample_laplacian;
    Eigen::SparseMatrix<double, Eigen::RowMajor> _sample_to_poly_codivergence;
    Eigen::SparseMatrix<double, Eigen::RowMajor> _sample_to_poly_gradient;
    Eigen::SparseMatrix<double, Eigen::RowMajor> _sample_codivergence;
    Eigen::SparseMatrix<double, Eigen::RowMajor> _sample_to_poly_dirichlet;
    Eigen::SparseMatrix<double, Eigen::RowMajor> _sample_to_poly_l2;
    std::vector<std::set<size_t>> _face_coboundary;
    mtao::VecXd _old_pressure_solution;
};
}  // namespace vem::fluidsim_3d
