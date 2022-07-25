#include <fmt/format.h>

#include <iostream>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <vem/from_polygons.hpp>
#include <vem/mesh.hpp>
#include <vem/monomial_cell_integrals.hpp>
#include <vem/polynomial_utils.hpp>
#include <vem/point_sample_indexer.hpp>

#include "vem/poisson_2d/poisson_vem.hpp"

void show_operator(const vem::poisson_2d::PoissonVEM2 &mesh) {
    auto c = mesh.get_cell(0);
    // auto G = c.G();
    // std::cout << "G: " << std::endl;
    // std::cout << G << std::endl;
    //// std::cout << "Determinant: " << G.determinant() << std::endl;
    //// std::cout << "Block Determinant: " <<
    //// G.bottomRightCorner(G.rows()-1,G.cols()-1).determinant() << std::endl;

    //// Eigen::EigenSolver<mtao::MatXd> eig(G);

    //// std::cout << "Eigenvalues: \n" << eig.eigenvalues().real().transpose()
    ///<< / std::endl; std::cout << "Eigenvalue determinant: " << /
    /// eig.eigenvalues().real().prod() << std::endl; std::cout <<
    /// "Eigenvectors: / \n" << eig.eigenvectors().eval() << std::endl;
    // std::cout << "B: " << std::endl;
    // auto B = c.B();
    // std::cout << B << std::endl;
    //// std::cout << "Determinant: " << B.determinant() << std::endl;
    // std::cout << "D: " << std::endl;
    auto D = c.D();
    // std::cout << D << std::endl;
    auto Pis = c.Pis();
    auto Pi = c.Pi();

    // std::cout << "We should have G = B * D, norm is " << (G - B * D).norm()
    //          << std::endl;
    // std::cout << "B * D\n" << (B * D) << std::endl;
    // std::cout << "Pis: \n" << Pis << std::endl;
    // std::cout << "DPis: \n" << Pis * D << std::endl;

    // auto C = c.C();
    // auto H = c.H();
    auto Pis0 = c.Pis0();
    auto Pi0 = c.Pi0();
    // auto M = c.M();
    // std::cout << "C: \n" << C << std::endl;
    // std::cout << "H: \n" << H << std::endl;
    // std::cout << "Pis0: \n" << Pis0 << std::endl;
    // std::cout << "DPis0: \n" << Pis0 * D << std::endl;

    // std::cout << "Diff\n" << Pis - Pis0 << std::endl;
    // std::cout << "M: \n" << M << std::endl;

    {
        auto Grad = mesh.poly_to_sample_gradient();
        std::cout << "Grad:\n"
                  << Grad << std::endl;
        auto M = mesh.mass_matrix();
        auto MM = mtao::eigen::sparse_block_diagonal_repmats(M, 2);
        std::cout << "MM:\n"
                  << MM << std::endl;
        auto coGrad = (Grad.transpose() * MM).eval();
        std::cout << "coGrad:\n"
                  << coGrad << std::endl;

        auto A = (coGrad * Grad).eval();
        std::cout << "Reconstructed lap:\n"
                  << A << std::endl;
        auto AA = mesh.poly_laplacian();
        std::cout << "My lap:\n"
                  << AA << std::endl;

        for (int j = 0; j < 20; ++j) {
            mtao::VecXd u = mtao::VecXd::Random(coGrad.cols());
            mtao::VecXd b = coGrad * u;

            mtao::VecXd x = b;
            x.setZero();
            mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-10);

            mtao::VecXd newu = u - Grad * x;
            std::cout << (coGrad * newu).transpose() << std::endl;
            std::cout << (coGrad * newu).norm() << std::endl;
        }

        // std::cout << "Poly mass matrix:\n" << c.H() << std::endl;
        // auto NMM = (Pis0.transpose() * c.H() * Pis0).eval();
        // std::cout << "Naive mass matrix implementation: \n" << NMM <<
        // std::endl;

        // std::cout << "Projection combination stuff" << std::endl;
        // std::cout << "Projection combination stuff" << std::endl;
        // std::cout << "Projection combination stuff" << std::endl;
        // std::cout << "Projection combination stuff" << std::endl;

        // std::cout << "Pi: \n" << Pi << std::endl;
        // std::cout << "Pi0: \n" << Pi0 << std::endl;
        // std::cout << "PiPi0:\n" << Pi * Pi0 << std::endl;
        // std::cout << "Pi0Pi:\n" << Pi0 * Pi << std::endl;
        // mtao::MatXd id = mtao::MatXd::Identity(Pi0.rows(), Pi0.cols()) - Pi0;
        // std::cout << "(1-Pi0) Pi0" << id * Pi0 << std::endl;
        // std::cout << "(1-Pi0) Pi" << id * Pi << std::endl;

        // std::cout << "Polky eval:\n" << D << std::endl;
        // std::cout << "Eval matrix:\n"
        //          <<
        //          mtao::MatXd(mesh.polynomial_to_sample_evaluation_matrix())
        //          << std::endl;
    }

    // std::cout << "Determinant: " << D.determinant() << std::endl;
    // std::cout << "Pi: " << std::endl;
    // auto Pi = c.Pi();
    // std::cout << Pi << std::endl;
    // std::cout << "Determinant: " << Pi.determinant() << std::endl;
    //
    {
        // auto gt = c.monomial_grammian();
        // auto pis = c.Pis();
        // auto pi = c.Pi();
        // auto Kpq = c.D();
        // auto MG = c.local_monomial_gradient();
        // auto M = c.monomial_l2_grammian();
        // auto MGx = MG.topRows(MG.rows() / 2);
        // auto MGy = MG.bottomRows(MG.rows() / 2);
        // double diam = std::pow<double>(c.diameter(), -2);
        // mtao::MatXd L = pis.transpose() * gt * pis;
        // auto id = mtao::MatXd::Identity(pi.rows(), pi.cols()) - pi;
        // mtao::MatXd E = id.transpose() * id;
        // std::cout << "E pass\n" << E * pi << std::endl;
        // std::cout << "Poly Poly stiffness\n" << L << std::endl;
        // std::cout << "Lap stiff part:\n" << L << std::endl;
        // std::cout << "err stiff part:\n" << E << std::endl;
        // std::cout << "full stiff:\n" << L + E << std::endl;
        // std::cout << "Monomial grammian:\n" << gt << std::endl;
        // std::cout << "Monomial grammian from c:\n"
        //          << (MGx.transpose() * M * MGx + MGy.transpose() * M * MGy)
        //          << std::endl;
        // mtao::MatXd idid(2 * c.system_size(), 2 * c.system_size());
        // idid.setConstant(0);
        // idid.topLeftCorner(id.rows(), id.cols()) = id;
        // idid.bottomRightCorner(id.rows(), id.cols()) = id;
        // std::cout << "IdId:\n" << idid << std::endl;
        // std::cout << "Cross term 1\n";
        // std::cout << "should be 0 to poly\n" << id * Kpq << std::endl;
        // std::cout << E * c.Grad_mOut() << std::endl << "\n";
        // std::cout << id.transpose() * c.Grad_mOut() << std::endl;
        // std::cout << "Cross term 2\n";
        // std::cout << "should be 0 to poly\n" << pis * id << std::endl;
        // std::cout << "should be 0 to poly\n" << id * pis << std::endl;
        // std::cout << c.CoGrad() * idid << std::endl;
        // std::cout << "Grad\n" << c.Grad() << std::endl;
        // std::cout << "CoGrad\n" << c.CoGrad() << std::endl;
        // std::cout << "Grad CoGrad prod\n" << c.CoGrad() * c.Grad() <<
        // std::endl; std::cout << "here's pis\n" << pis << std::endl; std::cout
        // << "Here's id:\n" << id << std::endl; std::cout << "Here's E :\n" <<
        // E
        // << std::endl; std::cout << "Pis id\n" << pis * id << std::endl;
        // std::cout << "Error idempotence:\n" << E * id << std::endl;
    }
    // std::cout << "Standard keh:\n" << c.KEH() << std::endl;
    // std::cout << "Product stiff:\n" << std::endl;
    // std::cout << c.CoGrad_mIn() * c.Grad_mOut() << std::endl;
    // std::cout << "Product keh:\n" << std::endl;
    // std::cout << c.CoGrad() * c.Grad() << std::endl;
}

void show_operators(const vem::VEMMesh2 &vem, int max_degree) {
    spdlog::info("max degree = {}", max_degree);
    show_operator(
      vem::poisson_2d::PoissonVEM2(vem, max_degree, max_degree - 1));
}

int main(int argc, char *argv[]) {
    mtao::vector<mtao::ColVecs2d> polys;
    auto &P = polys.emplace_back();
    P.resize(2, 4);
    P.col(0) << 0, 0;
    P.col(1) << 1, 0;
    P.col(2) << 1, 1;
    P.col(3) << 0, 1;

    auto vem = vem::from_polygons(polys);
    show_operators(vem, 1);
    show_operators(vem, 3);
    show_operators(vem, 4);

    return 0;
}
