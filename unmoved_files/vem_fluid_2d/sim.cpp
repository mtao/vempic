#include "sim.h"

#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>

#include "sim_vis.hpp"

Sim::~Sim() = default;
Sim::Sim(const mandoline::CutCellMesh<2>& ccm, const VEMMesh2& vem)
    : VEMMesh2FieldBase{ccm, vem},
      pressure(mtao::VecXd::Zero(vem.polynomial_size())),
      pressure_gradient(mtao::RowVecs2d::Zero(vem.polynomial_size(), 2)),
      velocity(mtao::RowVecs2d::Zero(vem.polynomial_size(), 2)),
      velocity_divergence(mtao::VecXd::Zero(vem.polynomial_size())) {}

void Sim::update_pressure_gradient() {
    pressure_gradient = vem.gradient(pressure);
}
void Sim::update_velocity_divergence() {
    velocity_divergence = vem.divergence(velocity);
}
VEMMesh2ScalarField Sim::scalar_field(const mtao::VecXd& u) const {
    assert(u.rows() == vem.polynomial_size());
    VEMMesh2ScalarField ret(*this);
    ret.coefficients = u;
    return ret;
}
VEMMesh2VectorField Sim::vector_field(const mtao::RowVecs2d& u) const {
    assert(u.rows() == vem.polynomial_size());
    VEMMesh2VectorField ret(*this);
    ret.coefficients = u;
    return ret;
}
VEMMesh2ScalarField Sim::pressure_field() const {
    return scalar_field(pressure);
}
VEMMesh2VectorField Sim::pressure_gradient_field() const {
    return vector_field(pressure_gradient);
}

VEMMesh2ScalarField Sim::velocity_divergence_field() const {
    return scalar_field(velocity_divergence);
}
VEMMesh2VectorField Sim::velocity_field() const {
    return vector_field(velocity);
}
void Sim::initialize_particles(
    size_t size, const std::function<mtao::Vec2d(const mtao::Vec2d&)>& vel) {
    particles.resize(4, size);
    for (size_t i = 0; i < size; ++i) {
        initialize_particle(i, vel);
    }
}
void Sim::initialize_particle(
    size_t i, const std::function<mtao::Vec2d(const mtao::Vec2d&)>& vel) {
    auto bb = ccm.bbox();
    auto p = particle_position(i) = bb.sample();
    particle_velocity(i) = vel(p);
}
void Sim::reinitialize_particles(size_t size) {
    particles.resize(4, size);
    auto vel = velocity_field();
    for (size_t i = 0; i < size; ++i) {
        initialize_particle(i, [&](const mtao::Vec2d& v) { return vel(v); });
    }
}
void Sim::update_particle_cell_cache() {
    particle_cell_cache.clear();
    particle_cell_cache.resize(vem.num_cells());
    for (int i = 0; i < particles.cols(); ++i) {
        auto p = particle_position(i);
        int cell = ccm.cell_index(p);
        if (cell >= 0 && cell < particle_cell_cache.size()) {
            particle_cell_cache[cell].emplace(i);
        } else {
            spdlog::warn("Particle {} at {} {} not cached", i, p.x(), p.y());
        }
    }
}
void Sim::step(double dt) {
    // TODO: cfl things
    int count = 1;
    double substep = dt / count;
    for (int j = 0; j < count; ++j) {
        update_particle_cell_cache();
        // move to the grid
        particle_velocities_to_field();

        // do the one grid operation
        pressure_projection();
        advect_particles_with_field(substep);
    }
}
void Sim::advect_particles_with_field(double dt) {
    auto vel = velocity_field();
    for (int pidx = 0; pidx < particles.cols(); ++pidx) {
        // rk2
        auto p = particle_position(pidx);
        mtao::Vec2d p2 = p + .5 * dt * vel(p);
        p = p + dt * vel(p2);
    }
}

void Sim::update_pressure() {
    std::map<size_t, double> dirichlet_vertices;
    std::map<size_t, double> neumann_edges;
    auto vfield = velocity_field();
    // find all the edges with neumann boundary conditions
    for (auto&& [ceidx, ce] : mtao::iterator::enumerate(ccm.cut_edges())) {
        if (ce.is_mesh_edge()) {
            auto a = ccm.vertex(ce.indices[0]);
            auto b = ccm.vertex(ce.indices[1]);
            auto N = ce.N;
            neumann_edges[ceidx] = N(0) * (b - a).norm();
        } else if (ce.external_boundary) {
            auto [c, sgn] = *ce.external_boundary;
            if (c < 0) {
                auto a = ccm.vertex(ce.indices[0]);
                auto b = ccm.vertex(ce.indices[1]);
                auto N = ce.N;
                mtao::Vec2d vel = mtao::Vec2d::Zero();
                auto indices = vem.edge_samples(ceidx);
                for (auto&& ind : indices) {
                    auto p = vem.sample_position(ind);
                    vel += vfield(p);
                }
                vel /= indices.size();

                neumann_edges[ceidx] = -N.dot(vel);
                // 0;  //(sgn ? -1 : 1) * N(0) * (b - a).norm();
            }
        }
    }
    for (auto&& [eidx, pr] :
         mtao::iterator::enumerate(ccm.exterior_grid.boundary_facet_pairs())) {
        auto [a, b] = pr;
        if (a < 0 || b < 0) {
            double sgn = b < 0 ? 1 : -1;
            int axis = ccm.exterior_grid.get_face_axis(eidx);
            mtao::Vec2d N = sgn * mtao::Vec2d::Unit(axis);

            mtao::Vec2d vel = mtao::Vec2d::Zero();
            auto indices = vem.edge_samples(eidx + ccm.num_cutedges());
            for (auto&& ind : indices) {
                std::cout << "Index: " << ind << " / " << vem.num_samples()
                          << std::endl;
                auto p = vem.sample_position(ind);
                vel += vfield(p);
            }
            vel /= indices.size();

            // neumann_edges[ceidx] = -N.dot(vel);

            neumann_edges[eidx + ccm.num_cutedges()] = N.dot(vel);  // 0;
            //(sgn ? -1 : 1) * N(0) * ccm.dx()(1 - axis);
        }
    }

    // std::vector<int> counts(vem.num_samples(), 0);
    // mtao::VecXd div(vem.num_samples());
    // div.setZero();
    // for (size_t i = 0; i < vem.num_cells(); ++i) {
    //    auto indices = vem.cell_sample_indices(i);
    //    size_t off = vem.coefficient_size();
    //    mtao::VecXd d = velocity_divergence.segment(i * off, off);

    //    for (auto&& si : indices) {
    //        counts[si]++;
    //        div(si) += vem.polynomial_eval(i, si, d)(0);
    //    }
    //}
    // for (auto&& [i, c] : mtao::iterator::enumerate(counts)) {
    //    if (c > 0) {
    //        div(i) / c;
    //    }
    //}
    mtao::VecXd u(velocity.size());
    {
        u.head(velocity.rows()) = velocity.col(0);
        u.tail(velocity.rows()) = velocity.col(1);

        auto BErr = vem.regression_error_bilinear();
        mtao::MatXd G = vem.gradient_sample2poly();
        mtao::MatXd D = vem.integrated_divergence_poly2adj_sample();
        mtao::MatXd L = vem.laplacian_sample2sample() + 100 * BErr;
        mtao::VecXd rhs = D * u;
        std::cout << "RHS poly error: " << (rhs.transpose() * BErr * rhs)
                  << std::endl;
        mtao::VecXd p = L.ldlt().solve(rhs);
        mtao::VecXd pg = G * p;
        mtao::VecXd up = u - pg;
        std::cout << "Residual div: " << (D * up).transpose().norm()
                  << "Even though u still has " << up.norm()
                  << ". Poly norm error: " << p.transpose() * BErr * p
                  << std::endl;
    }
    mtao::MatXd D = vem.integrated_divergence_poly2adj_sample();
    mtao::VecXd div = D * u;
    // mtao::VecXd vertex_values = vem.poisson_problem(div, {}, {});
    // dirichlet_vertices = {};
    neumann_edges = {};
    mtao::VecXd vertex_values =
        vem.poisson_problem(div, dirichlet_vertices, neumann_edges);
    Eigen::SparseMatrix<double> S2C = vem.sample2cell_coefficients();
    spdlog::warn("{}x{} {}", S2C.rows(), S2C.cols(), vertex_values.size());
    spdlog::warn("coeff {} * cells {} = {}; num_samples {}",
                 vem.coefficient_size(), vem.num_cells(),
                 vem.num_cells() * vem.coefficient_size(), vem.num_samples());
    {
        mtao::MatXd G = vem.gradient_sample2poly();
        mtao::MatXd D = vem.integrated_divergence_poly2adj_sample();
        auto BErr = vem.regression_error_bilinear();
        mtao::VecXd pg = G * vertex_values;
        mtao::VecXd up = u - pg;
        std::cout << "Residual div: " << (D * up).transpose().norm()
                  << "Even though u still has " << up.norm()
                  << ". Poly norm error: "
                  << vertex_values.transpose() * BErr * vertex_values
                  << std::endl;
        pressure = S2C * vertex_values;
        Eigen::SparseMatrix<double> S2C2 =
            mtao::eigen::sparse_block_diagonal_repmats(S2C, 2);
        spdlog::warn("s2c2 {} {} pg {}", S2C2.rows(), S2C2.cols(), pg.size());
        auto pgc = pg;
        pressure_gradient.resize(vem.coefficient_size() * vem.num_cells(), 2);
        pressure_gradient.col(0) = pgc.head(pressure_gradient.rows());
        pressure_gradient.col(1) = pgc.tail(pressure_gradient.rows());

        spdlog::warn("{} {} up {}", S2C2.rows(), S2C2.cols(), up.size());
        auto vel = up;
        velocity.col(0) = vel.head(velocity.rows());
        velocity.col(1) = vel.tail(velocity.rows());
    }
}
void Sim::pressure_projection() {
    update_velocity_divergence();

    update_pressure();
    // update_pressure_gradient();

    // project particle velocities from teh smooth field
    auto pg = pressure_gradient_field();
    auto vel = velocity_field();
    for (int pidx = 0; pidx < particles.cols(); ++pidx) {
        auto p = particle_position(pidx);
        auto v = particle_velocity(pidx);
        v = vel(p);
    }
    // std::cout << "Velocity" << std::endl;
    // std::cout << velocity << std::endl;
    // std::cout << "pressure_gradient" << std::endl;
    // std::cout << pressure_gradient << std::endl;
    // velocity -= pressure_gradient;
    // std::cout << "Post-pressure divergence " <<
    // vem.divergence(velocity).norm()
    //          << std::endl;
}
void Sim::particle_velocities_to_field() {
    velocity.resize(vem.polynomial_size(), 2);

    size_t off = vem.coefficient_size();
    auto X = velocity.col(0);
    auto Y = velocity.col(1);
    for (auto&& [c, ps] : mtao::iterator::enumerate(particle_cell_cache)) {
        auto x = X.segment(c * off, off);
        auto y = Y.segment(c * off, off);
        if (ps.size() < 1) {
            x.setConstant(0);
            y.setConstant(0);
        } else {
            std::vector<int> indices(ps.begin(), ps.end());
            mtao::RowVecs2d V(indices.size(), 2);
            for (auto&& [idx, ind] : mtao::iterator::enumerate(indices)) {
                V.row(idx) = particle_velocity(ind).transpose();
            }

            mtao::MatXd coeff_mat(vem.coefficient_size(), indices.size());

            for (auto&& [idx, ind] : mtao::iterator::enumerate(indices)) {
                auto p = particle_position(ind);
                mtao::VecXd pe = vem.polynomial_entries(c, p).transpose();
                coeff_mat.col(idx) = pe;
            }

            mtao::MatXd m = coeff_mat.transpose() * coeff_mat;

            auto ldlt = m.ldlt();
            x = coeff_mat * ldlt.solve(V.col(0));
            y = coeff_mat * ldlt.solve(V.col(1));
        }
    }
}
