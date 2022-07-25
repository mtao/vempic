#include <mtao/geometry/mesh/laplacian.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include "dirichlet_triangle_laplacian.h"
#include <mtao/iterator/enumerate.hpp>


mtao::VecXd dirichlet_laplacian(const mtao::ColVecs2d& V, const mtao::ColVecs3i& F, const std::map<size_t,double>& dv) {
    Eigen::SparseMatrix<double> L = mtao::geometry::mesh::cot_laplacian(V,F);
    //auto trips = mtao::eigen::mat_to_triplets(L);
    //std::cout << L << std::endl;
    size_t off = L.rows();
    L.conservativeResize(off + dv.size(), L.cols() + dv.size());
    mtao::VecXd rhs(off+dv.size());
    rhs.setZero();
    for(auto&& [idx,pr]: mtao::iterator::enumerate(dv)) {
        auto [i,v] = pr;
        L.coeffRef(off+idx,i) = 1;
        L.coeffRef(i,off+idx) = 1;
        rhs(off+idx) = v;
    }

    mtao::VecXd x(off+dv.size());
    x.setZero();


    mtao::solvers::linear::CholeskyPCGSolve(L,rhs,x);
    return x.topRows(off);
}

