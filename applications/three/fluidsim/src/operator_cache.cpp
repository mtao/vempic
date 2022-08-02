#include "vem/three/fluidsim/operator_cache.hpp"
#include <mtao/logging/stopwatch.hpp>

#include <chrono>
#include <thread>
#include <vem/three/boundary_facets.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "mtao/eigen/sparse_block_diagonal_repmats.hpp"

namespace vem::three::fluidsim {
OperatorCache::OperatorCache(const FluidVEM3& fvem, bool build) : _fvem(fvem) {
    if (build) {
        build_cache();
    }
}
void OperatorCache::build_cache() {
    _built = true;
    auto sw = mtao::logging::hierarchical_stopwatch("operators");

    spdlog::info("Building operator cache with {} active cells",
                 _fvem.active_cells().size());

    const auto poly_v2p = _fvem.velocity_stride_to_pressure_monomial_map();
    spdlog::info("Finished poly v2p ({} vel monomials, {} pressure monomials",
                 poly_v2p.cols(), poly_v2p.rows());
    const auto G = _fvem.pressure_indexer().monomial_indexer().gradient();
    spdlog::info("Finished poly grad");

    _sample_to_poly_dirichlet = _fvem.sample_to_poly_dirichlet();
    spdlog::info("Finished Building sample2poly dirichlet");
    _sample_to_poly_l2 = _fvem.sample_to_poly_l2();
    spdlog::info("Finished Building sample2poly l2");
    {
        const auto& dPis = sample_to_poly_dirichlet();
        //// auto gPis = sample_to_poly_l2();

        _sample_to_poly_gradient = (mtao::eigen::sparse_block_diagonal_repmats(
                                        poly_v2p.transpose().eval(), 3) *
                                    G * dPis)
                                       .eval();
        spdlog::info("Finished Building sample2poly gradient {}x{}",
                     _sample_to_poly_gradient.rows(),
                     _sample_to_poly_gradient.cols());
    }

    auto M = _fvem.poly_velocity_l2_grammian();
    auto M3 = mtao::eigen::sparse_block_diagonal_repmats(M, 3);
    spdlog::info("Monomial L2^3 shape: {}x{}", M3.rows(), M3.cols());

    //_sample_laplacian =
    //    _sample_to_poly_gradient.transpose() * M3 * _sample_to_poly_gradient;

    //using namespace std::chrono_literals;
    //spdlog::info("Taking a 5s nap");
    //std::this_thread::sleep_for(5s);
    _sample_laplacian = _fvem.sample_laplacian();
    //spdlog::info("Taking a 30s nap");
    //std::this_thread::sleep_for(30s);
    // Eigen::SparseMatrix<double, Eigen::RowMajor> L2 =
    // _fvem.sample_laplacian(); spdlog::warn("Laplacian error: {}",
    // (_sample_laplacian - L2).norm()); _sample_laplacian =
    //_fvem.sample_laplacian();
    spdlog::info("Finished Building Laplacian");

    _old_pressure_solution = mtao::VecXd::Zero(_sample_laplacian.rows());

    int nonzeros = _sample_laplacian.nonZeros();
    _sample_laplacian.prune(0.0, 1e-5);
    spdlog::info(
        "Through trivial pruning we obtain {0} nnz of {2} entries "
        "rather than {1} nnz of {2} entries",
        _sample_laplacian.nonZeros(), nonzeros, _sample_laplacian.size());

    {
        // const auto& l2Pis = sample_to_poly_l2();
        // auto l2G = _fvem.poly_pressure_l2_grammian();
        // spdlog::info("Finished pressure l2 grammian");
        // Eigen::SparseMatrix<double> VS = l2G * poly_v2p * l2Pis;
        // auto VSVS = mtao::eigen::sparse_block_diagonal_repmats(VS, 3);

        //_sample_to_poly_codivergence = G.transpose() * VSVS;
        // spdlog::info("Finished Building sample2poly Codivergence {}x{}",
        //             _sample_to_poly_codivergence.rows(),
        //             _sample_to_poly_codivergence.cols());
    }

    {
        const auto& P = sample_to_poly_l2();

        const auto& SPGrad = sample_to_poly_gradient();
        _sample_codivergence = SPGrad.transpose() * M3 *
                               mtao::eigen::sparse_block_diagonal_repmats(P, 3);
        spdlog::info("Finished Building Codivergence {}x{}",
                     _sample_codivergence.rows(), _sample_codivergence.cols());
        // const auto& P = sample_to_poly_dirichlet();

        // const auto& SPCod = sample_to_poly_codivergence();
        //_sample_codivergence = P.transpose() * SPCod;
        // spdlog::info("Finished Building Codivergence {}x{}",
        //             _sample_codivergence.rows(),
        //             _sample_codivergence.cols());
        // std::cout << (_fvem.sample_codivergence().toDense() -
        //              _sample_codivergence)
        //                 .norm()
        //          << std::endl;
    }

    // std::cout << (_fvem.sample_gradient().toDense() -
    // _sample_gradient).norm()
    //          << std::endl;
    // std::cout << _fvem.sample_gradient().norm() << " "
    //          << _sample_gradient.norm() << std::endl;
    // codivergence
    //
    //

    _face_coboundary =
        face_coboundary_map(_fvem.mesh(), _fvem.active_cells());

    spdlog::info("Done building operator cache");
    // invert things in the pressure solution
    /*
     Eigen::PermutationMatrix<Eigen::Dynamic>
     P(_old_pressure_solution.rows());

     for (int id_old_pressure_solution = 0;
         id_old_pressure_solution < _old_pressure_solution.rows();
         ++id_old_pressure_solution) {
        P.indices()(id_old_pressure_solution) =
            _old_pressure_solution.rows() - id_old_pressure_solution - 1;
    }
    _sample_laplacian = P * _sample_laplacian * P;
    _sample_codivergence =  P * _sample_codivergence;
    _sample_to_poly_dirichlet=  _sample_to_poly_dirichlet * P;
    _sample_gradient =  _sample_gradient * P;
    _sample_codivergence =  _sample_codivergence * P;
    */

    // welp
    // exit(0);
}
}  // namespace vem::fluidsim_3d
