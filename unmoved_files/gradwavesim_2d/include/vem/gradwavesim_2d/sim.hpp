#pragma once
#include <vem/monomial_field_embedder.hpp>
#include <vem/poisson_2d/poisson_vem.hpp>

namespace vem::gradwavesim_2d {

class Sim {
  public:
    Sim(const VEMMesh2 &vem, int order);

    const VEMMesh2 &mesh() const { return poisson_vem.mesh(); }

    void step(double dt);

    void implicit_stormer_verlet_update(double dt);

    void initialize();

    size_t pressure_sample_count() const;
    size_t pressure_polynomial_count() const;

    size_t vector_field_sample_count() const;
    size_t vector_field_polynomial_count() const;

    // (sample)vector field + (monomial)pressure lambdas
    size_t system_size() const;
    std::array<size_t, 4> kkt_block_offsets() const;


    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(double dt) const;

    double c = .1;
    // domain definition stuff
    std::set<int> active_cells;
    poisson_2d::ScalarConstraints boundary_conditions;

    const vem::poisson_2d::PoissonVEM2 poisson_vem;
    // samples
    mtao::VecXd gradpressure;
    mtao::VecXd gradpressure_previous;
    mtao::VecXd gradpressure_dtdt;

    // monomials
    mtao::VecXd pressure;
};
}// namespace vem::gradwavesim_2d
