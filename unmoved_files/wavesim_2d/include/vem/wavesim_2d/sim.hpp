#pragma once
#include <vem/monomial_field_embedder.hpp>
#include <vem/poisson_2d/poisson_vem.hpp>
#include <vem/serialization/prioritizing_inventory.hpp>

namespace vem::wavesim_2d {

class Sim {
  public:
    Sim(const VEMMesh2 &vem, int order, serialization::Inventory* parent = nullptr);

    const VEMMesh2 &mesh() const { return poisson_vem.mesh(); }

    void step(double dt);

    void implicit_stormer_verlet_update(double dt);
    void explicit_stormer_verlet_integration(double dt);

    void initialize();

    size_t pressure_sample_count() const;
    size_t pressure_polynomial_count() const;

    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(double dt) const;

    double c = .1;
    // domain definition stuff
    std::set<int> active_cells;
    poisson_2d::ScalarConstraints boundary_conditions;

    const vem::poisson_2d::PoissonVEM2 poisson_vem;
    mtao::VecXd pressure;
    mtao::VecXd pressure_previous;
    mtao::VecXd pressure_dtdt;

    serialization::PrioritizingInventory inventory;
    int frame_index = 0;
    void initialize_inventory();
};
}// namespace vem::wavesim_2d
