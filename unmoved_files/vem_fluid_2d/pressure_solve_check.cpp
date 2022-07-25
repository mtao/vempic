#include <mandoline/construction/construct2.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>

#include "cutmesh2_to_vemmesh.hpp"
#include "pointwise_vem_mesh2.hpp"

std::tuple<mandoline::CutCellMesh<2>, PointwiseVEMMesh2> get_vem(
    int N = 5, int order = 2, int edge_counts = 1) {
    mandoline::CutCellMesh<2> ccm =
        mandoline::construction::from_grid({}, {}, {{N, N}});
    PointwiseVEMMesh2 vem;
    cutmesh2_to_vemmesh(ccm, vem);
    vem.desired_edge_counts = edge_counts;
    vem.projection_lambda = 10;
    vem.order = order;
    vem.initialize_interior_offsets();
    return {ccm, vem};
}

// \nabla p = u;
// \nabla^* \nabla p = \nabla^* u;

mtao::VecXd input_pointwise_exact(const PointwiseVEMMesh2& vem) {
    mtao::VecXd R(vem.num_samples());
    for (int sample_index = 0; sample_index < vem.num_samples();
         ++sample_index) {
        auto v = vem.sample_position(sample_index);
        R(sample_index) =
            std::pow<double>((v - mtao::Vec2d::Constant(.5)).norm(), 2.);
        // R(vem.num_samples() + sample_index) = v.y();
    }
    return R;
}
mtao::VecXd input_polynomial_exact(const PointwiseVEMMesh2& vem) {
    Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
    return S2C * input_pointwise_exact(vem);
}

mtao::VecXd input_polynomial_coexact(const PointwiseVEMMesh2& vem) {
    Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
    return S2C * input_pointwise_exact(vem);
}
mtao::VecXd input_pointwise_coexact(const PointwiseVEMMesh2& vem) {
    Eigen::SparseMatrix<double> C2S = vem.poly2sample();
    return C2S * input_polynomial_coexact(vem);
}

Eigen::SparseMatrix<double> poly2poly_gradient(const PointwiseVEMMesh2& vem) {
    return vem.gradient();
}
Eigen::SparseMatrix<double> poly2sample_gradient(const PointwiseVEMMesh2& vem) {
    Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
    return poly2poly_gradient(vem) * S2C;
}

mtao::VecXd rhs_from_pointwise(const PointwiseVEMMesh2& vem) {
    return poly2sample_gradient(vem).transpose() * input_pointwise_coexact(vem);
}
void solve(const PointwiseVEMMesh2& vem) {
    auto G = poly2sample_gradient(vem);
    Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
    mtao::VecXd L = G.transpose() * G * S2C * rhs_from_pointwise(vem);
}

void test_polynomial_laplacian() {
    constexpr static int mode = 2;
    int row_count = 3;
    if constexpr (mode == 0) {
        row_count = 2;
    } else if constexpr (mode == 1) {
        row_count = 3;

    } else if constexpr (mode == 2) {
        row_count = 5;
    }
    auto [ccm, vem] = get_vem(row_count);
    // auto dx = vem.poly_dx();
    // auto dy = vem.poly_dy();

    // std::cout << "dx2\n";
    // std::cout << dx.transpose() * dx << std::endl;
    // std::cout << "dx2" << std::endl;
    // std::cout << dx* dx << std::endl;
    // std::cout << "dy2\n";
    // std::cout << dy.transpose() * dy << std::endl;
    // std::cout << "dy2" << std::endl;
    // std::cout << dy* dy << std::endl;

    // std::cout << "nabla\n";
    // std::cout << vem.poly_gradient() << std::endl;
    // auto G = vem.divergence();

    ////mtao::MatXd L = dx.transpose() * MI.asDiagonal() * dx +
    ////    dy.transpose() * MI.asDiagonal() * dy;
    // mtao::MatXd L = dx.transpose() * dx +
    //    dy.transpose()  * dy;
    // std::cout << "Laplacian:\n";
    // std::cout << L << std::endl;
    // mtao::MatXd IL = L.inverse();
    // IL.row(0).setZero();
    // std::cout << "Laplacian inverse:\n";
    // std::cout << IL << std::endl;
    // auto GS = dx.transpose() + dy.transpose();

    int poly_size = vem.num_cells() * vem.coefficient_size();

    if constexpr (mode == 0) {
        // G p = u
        // D G p = D u
        auto MI = vem.monomial_integrals();
        mtao::MatXd P = mtao::MatXd::Identity(poly_size, poly_size);
        mtao::MatXd G = vem.poly_gradient();
        // mtao::MatXd rhs = D * U;

        auto MI2 = mtao::eigen::vstack(MI, MI);
        std::cout << "Mass matrix: " << MI.transpose() << std::endl;

        mtao::MatXd D = G.transpose() * MI2.asDiagonal();

        mtao::MatXd L = D * G;
        std::cout << "D\n" << D << std::endl;
        std::cout << "G\n" << G << std::endl;
        mtao::MatXd U = G;
        std::cout << "U\n" << U << std::endl;
        std::cout << "GU\n" << L.ldlt().solve(G * U) << std::endl;

        std::cout << "P\n";
        std::cout << P << std::endl;
        std::cout << "Lap:\n";
        std::cout << L << std::endl;

        // mtao::MatXd u = mtao::VecXd::Constant(1,2 * poly_size).asDiagonal();

        // mtao::VecXd u = mtao::VecXd::Random(2 * poly_size);
        mtao::VecXd u = mtao::VecXd::Random(poly_size);
        mtao::VecXd rhs = D * u;
        std::cout << "Thing that should be identity" << std::endl;
        std::cout << (G * (L).ldlt().solve(D)) << std::endl;
        mtao::VecXd p = L.ldlt().solve(rhs);
        std::cout << "pressure space residual: " << (L * p - rhs).norm()
                  << std::endl;
        mtao::VecXd pg = G * p;
        mtao::VecXd up = u - pg;
        std::cout << "Residual div: " << (D * up).transpose() << std::endl;
    } else if constexpr (mode == 1) {
        // G p = u
        // D G p = D u
        auto MI = vem.monomial_integrals();
        mtao::MatXd P = mtao::MatXd::Identity(poly_size, poly_size);
        mtao::MatXd G = vem.gradient();
        // mtao::MatXd rhs = D * U;

        // std::cout << "Mass matrix: " << MI.transpose() << std::endl;
        auto MI2 = mtao::eigen::vstack(MI, MI);

        mtao::MatXd D = vem.gradient().transpose() * MI2.asDiagonal();
        mtao::MatXd L = D * G;

        // std::cout << "D\n" << D << std::endl;
        // std::cout << "G\n" << G << std::endl;
        //    std::cout << "P\n";
        // std::cout << P << std::endl;
        // std::cout << "Lap:\n";
        // std::cout << D * G << std::endl;

        mtao::VecXd u = mtao::VecXd::Random(2 * poly_size);
        mtao::VecXd rhs = D * u;
        mtao::VecXd p = L.ldlt().solve(rhs);
        mtao::VecXd pg = G * p;
        mtao::VecXd up = u - pg;
        std::cout << "Residual div: " << (D * up).transpose().norm()
                  << "Even though u still has " << up.norm() << std::endl;
    } else if constexpr (mode == 2) {
        // G p = u
        // D G p = D u
        auto BErr = vem.regression_error_bilinear();
        mtao::MatXd G = vem.gradient_sample2poly();
        mtao::MatXd D = vem.integrated_divergence_poly2adj_sample();
        mtao::MatXd L = vem.laplacian_sample2sample();  // + 100 * BErr;

        std::cout << "Bilinear term: \n" << BErr << std::endl;
        Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
            qr(S2C);
        std::cout << "Q\n" << mtao::MatXd(qr.matrixQ()) << std::endl;
        std::cout << "R\n" << qr.matrixR() << std::endl;
        spdlog::warn("QR shape: {}x{} rank {}", qr.rows(), qr.cols(),
                     qr.rank());

        mtao::VecXd u = mtao::VecXd::Random(2 * poly_size);
        mtao::VecXd rhs = D * u;
        std::cout << "RHS poly error: " << (rhs.transpose() * BErr * rhs)
                  << std::endl;
        {
            mtao::VecXd p = L.ldlt().solve(rhs);

            mtao::VecXd pg = G * p;
            mtao::VecXd up = u - pg;

            std::cout << "Residual div: " << (D * up).transpose().norm()
                      << "Even though u still has " << up.norm()
                      << ". Poly norm error: " << p.transpose() * BErr * p
                      << std::endl;
        }
        {
            mtao::VecXd p = rhs;
            p.setZero();

            mtao::solvers::linear::CholeskyPCGSolve(L, rhs, p);
            mtao::VecXd pg = G * p;
            mtao::VecXd up = u - pg;

            std::cout << "Residual div: " << (D * up).transpose().norm()
                      << "Even though u still has " << up.norm()
                      << ". Poly norm error: " << p.transpose() * BErr * p
                      << std::endl;
        }
        {
            mtao::VecXd sp = vem.poisson_problem(rhs);
            spdlog::warn("S2C: {} {}; sp: {}", S2C.rows(), S2C.cols(),
                         sp.size());
            mtao::VecXd p = S2C * sp;

            // mtao::solvers::linear::CholeskyPCGSolve(L, rhs, p);
            spdlog::warn("G : {} {}; p: {}", G.rows(), G.cols(), p.size());
            mtao::VecXd pg = G * sp;
            spdlog::warn("u: {}; pg: {}", u.size(), pg.size());
            mtao::VecXd up = u - pg;
            spdlog::warn("D : {} {}; up: {}", D.rows(), D.cols(), up.size());

            std::cout << "Residual div: " << (D * up).transpose().norm()
                      << std::endl;
            std::cout << "Even though u still has " << up.norm()
                      << ". Poly norm error: " << sp.transpose() * BErr * sp
                      << std::endl;
        }
    } else if constexpr (mode == 3) {
        Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();

        auto BErr = vem.regression_error_bilinear();
        spdlog::warn("S2C {}x{}", S2C.rows(), S2C.cols());
        spdlog::warn("BErr {}x{}", BErr.rows(), BErr.cols());
        std::cout << S2C * BErr << std::endl;

        std::cout << " Going through cells now " << std::endl;
        for (int j = 0; j < vem.num_cells(); ++j) {
            std::cout << "Cell " << j << std::endl;
            auto P = vem.poly_projection_sample2poly(j);
            auto N = vem.poly_projection_kernel(j);
            // std::cout << "Proj:\n" << P << std::endl;
            // std::cout << "Kern:\n" << N << std::endl;
            // std::cout << "Err:\n" << P * N << std::endl;
        }
    }
    std::cout << "Mode was " << mode << std::endl;

    // std::cout << vem.gradient() * IL * GS << std::endl;

    // std::cout << "Gradient: " << std::endl;
    // std::cout << G << std::endl;
    // std::cout << G  * G.transpose() << std::endl;
    // std::cout << "Mass matrix: " << MI.transpose() << std::endl;
    // Eigen::SparseMatrix<double> L = G  * G.transpose();
    // Eigen::SparseMatrix<double> L = G * MI2.asDiagonal() * G.transpose();
    // std::cout << L << std::endl;
}

void test_least_squares_gradient() {
    auto [ccm, vem] = get_vem();
    auto grid = ccm.vertex_grid();
    auto grid_shape = grid.shape();
    int max = grid_shape[1] - 1;
    auto G = vem.integrated_divergence_sample2sample();
    std::map<size_t, double> dirichlet_vertices;
    std::map<size_t, double> neumann_edges;
    dirichlet_vertices[grid.index(0, 0)] = 1;

    Eigen::SparseMatrix<double> C2S = vem.poly2sample();
    Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
    Eigen::SparseMatrix<double> S2C2 =
        mtao::eigen::sparse_block_diagonal_repmats(S2C, 2);
}

int main(int argc, char* argv[]) {
    test_polynomial_laplacian();
#ifdef UNUSED
    auto grid = ccm.vertex_grid();
    auto grid_shape = grid.shape();
    int max = grid_shape[1] - 1;

    /*
    dirichlet_vertices[grid.index(grid_shape[0] - 1, grid_shape[1] - 1)] = -1;

    auto v = vem.laplace_problem(dirichlet_vertices, neumann_edges);
    */
    mtao::VecXd R(2 * vem.num_samples());
    for (int sample_index = 0; sample_index < vem.num_samples();
         ++sample_index) {
        auto v = vem.sample_position(sample_index);
        R(sample_index) = v.x();
        R(vem.num_samples() + sample_index) = v.y();
    }
    std::cout << "X" << std::endl;
    for (int j = 0; j < grid_shape[1]; ++j) {
        for (int i = 0; i < grid_shape[0]; ++i) {
            fmt::print("{:02.4f} ", R(grid.index(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << "Y" << std::endl;
    for (int j = 0; j < grid_shape[1]; ++j) {
        for (int i = 0; i < grid_shape[0]; ++i) {
            fmt::print("{:02.4f} ", R(vem.num_samples() + grid.index(i, j)));
        }
        std::cout << std::endl;
    }

    std::cout << "Coefficients: " << std::endl;
    std::cout << (S2C2 * R).transpose() << std::endl;
    {
        auto v = S2C2 * R;
        std::cout << "x" << std::endl;
        std::cout
            << v.head(vem.num_cells() * vem.coefficient_size()).transpose()
            << std::endl;
        std::cout << "y" << std::endl;
        std::cout
            << v.tail(vem.num_cells() * vem.coefficient_size()).transpose()
            << std::endl;
    }

    std::cout << "Divergence: \n";
    std::cout << (vem.divergence() * S2C2 * R).transpose() << std::endl;
    mtao::VecXd div = vem.divergence_sample2poly() * R;

    if (false) {
        auto G = vem.gradient_sample2poly();
        auto I = vem.monomial_integrals();
        auto S2C = vem.sample2cell_coefficients();
        Eigen::SparseMatrix<double> S2C2 =
            mtao::eigen::sparse_block_diagonal_repmats(S2C, 2);
        mtao::VecXd I2 = mtao::eigen::vstack(I, I);
        std::cout << "gradient sample to cell: " << std::endl;
        std::cout << (S2C2 * R).transpose() << std::endl;
        ;
        std::cout << "Copied twice: " << std::endl;
        std::cout << (I2.asDiagonal() * S2C2 * R).transpose() << std::endl;
        ;
        std::cout << "Final rhs: " << std::endl;
        ;
        std::cout << (G.transpose() * I2.asDiagonal() * S2C2 * R).transpose()
                  << std::endl;
        ;
    }
    // std::cout << C2S << std::endl;
    // std::cout << "position divergences" << std::endl;
    // std::cout << (vem.poly2sample() * div).transpose() << std::endl;

    mtao::VecXd rhs = vem.integrated_divergence_sample2adj_sample() * R;
    // std::cout << "rhs: " << std::endl;
    // std::cout << rhs.transpose() << std::endl;

    for (int j = 0; j < grid_shape[1]; ++j) {
        for (int i = 0; i < grid_shape[0]; ++i) {
            fmt::print("{:02.4f} ", rhs(grid.index(i, j)));
        }
        std::cout << std::endl;
    }
    {
        mtao::VecXd rhs = vem.integrated_divergence_sample2sample() * R;
        std::cout << "rhs: " << rhs.transpose() << std::endl;
        std::cout << vem.monomial_integrals().rows() << std::endl;
        auto L = vem.laplacian();
        // std::cout << "L: " << L << std::endl;

        auto LL = vem.laplacian_sample2sample();
        auto G = vem.gradient_sample2poly();
        mtao::VecXd P = mtao::VecXd::Random(vem.num_samples());
        P = R.head(vem.num_samples());
        mtao::VecXd PG = G * P;
        auto I = vem.monomial_integrals();
        mtao::VecXd I2 = mtao::eigen::vstack(I, I);
        mtao::VecXd true_rhs = G.transpose() * I2.asDiagonal() * PG;
        std::cout << "True rhs: " << std::endl;
        std::cout << true_rhs.transpose() << std::endl;
        spdlog::warn("Our rhs: {}x{} {}", LL.rows(), LL.cols(), P.size());
        std::cout << (LL * P).transpose() << std::endl;
        std::cout << "Current rhs: " << std::endl;
        std::cout << rhs.transpose() << std::endl;
        return 0;

        // std::cout << "LL: \n" << LL << std::endl;
        std::cout << "L leakage: "
                  << mtao::VecXd::Ones(vem.num_samples()).transpose() * LL *
                         mtao::VecXd::Ones(vem.num_samples())
                  << std::endl;
        LL = (LL.array().abs() > 1e-5).select(LL, 0);
        spdlog::warn("Samples {} cells {}", vem.num_samples(), vem.num_cells());
        std::cout << "LL: \n" << LL << std::endl;
        auto solver = LL.ldlt();
        if (solver.info() != Eigen::Success) {
            spdlog::error("LDLT failed to compute!");
        }
        spdlog::warn("{}x{} {}", LL.rows(), LL.cols(), rhs.size());
        mtao::VecXd sol = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            spdlog::error("LDLT failed to solve!");
        }
        std::cout << "poly poly solution" << std::endl;
        std::cout << sol.transpose() << std::endl;
    }
    // auto L = vem.laplacian({});
    // fmt::print("System A{}x{} x{} = b{}", L.rows(), L.cols(), L.cols(),
    //           rhs.size());
    // mtao::VecXd v = vem.poisson_problem(rhs, dirichlet_vertices,
    // neumann_edges);
    mtao::VecXd v = vem.laplace_problem(dirichlet_vertices, neumann_edges);
    std::cout << v.transpose() << std::endl;
    for (int j = 0; j < grid_shape[1]; ++j) {
        for (int i = 0; i < grid_shape[0]; ++i) {
            fmt::print("{:02.4f} ", v(grid.index(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout
        << (vem.gradient() * vem.sample2cell_coefficients() * v).transpose()
        << std::endl;

    {
        Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();

        // because the divergence is two dimensions stacked we need to create
        // the block matrix that undoes it
        mtao::VecXd h =
            S2C2 * vem.gradient() * vem.sample2cell_coefficients() * v;
        std::cout << "x" << std::endl;
        std::cout
            << h.head(vem.num_cells() * vem.coefficient_size()).transpose()
            << std::endl;
        std::cout << "y" << std::endl;
        std::cout
            << h.tail(vem.num_cells() * vem.coefficient_size()).transpose()
            << std::endl;
    }
#endif
}
