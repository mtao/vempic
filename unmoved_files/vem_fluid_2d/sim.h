#pragma once
#include <mandoline/mesh2.hpp>

#include "vem_mesh2.hpp"
#include "vem_mesh_interpolation2.hpp"

class SimVis;

class Sim : public VEMMesh2FieldBase {
   public:
    friend class SimVis;
    Sim(const mandoline::CutCellMesh<2>& ccm, const VEMMesh2& vem);
    ~Sim();

    VEMMesh2ScalarField scalar_field(const mtao::VecXd& u) const;
    VEMMesh2VectorField vector_field(const mtao::RowVecs2d& u) const;

    VEMMesh2ScalarField pressure_field() const;
    VEMMesh2VectorField pressure_gradient_field() const;

    VEMMesh2ScalarField velocity_divergence_field() const;
    VEMMesh2VectorField velocity_field() const;

    int num_particles() const { return particles.cols(); }
    void initialize_particles(
        size_t size,
        const std::function<mtao::Vec2d(const mtao::Vec2d&)>& velocity =
            [](const mtao::Vec2d& v) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        });
    void initialize_particle(
        size_t index,
        const std::function<mtao::Vec2d(const mtao::Vec2d&)>& velocity =
            [](const mtao::Vec2d& v) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        });
    // intiializes particles using the current velocity field
    void reinitialize_particles(size_t size);
    mtao::ColVecs4d particles;
    auto particle_positions() { return particles.topRows<2>(); }
    auto particle_velocities() { return particles.bottomRows<2>(); }
    auto particle_positions() const { return particles.topRows<2>(); }
    auto particle_velocities() const { return particles.bottomRows<2>(); }
    auto particle_position(size_t i) { return particle_positions().col(i); }
    auto particle_velocity(size_t i) { return particle_velocities().col(i); }
    auto particle_position(size_t i) const {
        return particle_positions().col(i);
    }
    auto particle_velocity(size_t i) const {
        return particle_velocities().col(i);
    }

    void step(double dt);

    void advect_particles_with_field(double dt);
    void particle_velocities_to_field();
    void update_particle_cell_cache();

    void pressure_projection();
    void update_pressure();             // from divergence
    void update_pressure_gradient();    // from pressure
    void update_velocity_divergence();  // from velocity field

    mtao::ColVecs2d particle_field_velocities();

    mtao::VecXd pressure;
    mtao::RowVecs2d pressure_gradient;
    mtao::RowVecs2d velocity;
    mtao::VecXd velocity_divergence;

    std::vector<std::set<int>> particle_cell_cache;
};

