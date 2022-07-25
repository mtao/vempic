#pragma once
#include <Eigen/CholmodSupport>
#include <Eigen/SPQRSupport>
#include <nlohmann/json.hpp>
#include <vem/monomial_field_embedder3.hpp>
#include <vem/serialization/inventory.hpp>
#include <vem/serialization/prioritizing_inventory.hpp>
#include <vem/wavesim_3d/wavevem3.hpp>

namespace vem::wavesim_3d {

class Sim : public WaveVEM3 {
   public:
    Sim(const VEMMesh3& vem, int degree,
        std::shared_ptr<serialization::Inventory> inventory = nullptr);
    void initialize(const std::function<double(const mtao::Vec3d&)>& f);

    void step(double dt);
    void semiimplicit_step(double dt);
    void update_polynomial_pressure();

    size_t system_size() const;

    void set_active_cells(std::set<int> c) override;

    // polynomial pressure
    mtao::VecXd pressure;
    mtao::VecXd sample_pressure;
    mtao::ColVecs3d pressure_gradient;

    std::shared_ptr<serialization::Inventory> inventory;
    // yes yes, this is unsafe, but i don't want to mess with the prioritizing
    // inventory stuff at the moment
    serialization::Inventory* active_inventory = nullptr;
    // std::shared_ptr<serialization::PrioritizingInventoryHandler>
    // inventory_handler;
    int frame_index = 0;

    void initialize_inventory();

    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        _qr_solver;

    Eigen::SPQR<Eigen::SparseMatrix<double>> _spqr_solver;

    bool solver_warm = false;
    Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>> _cholmod_solver;
    bool force_static_stiffness_reconstruction = true;
};
}  // namespace vem::wavesim_3d
