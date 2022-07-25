#define EIGEN_DONT_PARALLELIZE
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/logging/stopwatch.hpp>
#include <vem/utils/boundary_facets.hpp>
#include <vem/utils/dehomogenize_vector_points.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "vem/fluidsim_3d/sim.hpp"
#include "vem/utils/coefficient_accumulator3.hpp"
namespace vem::fluidsim_3d {
// void Sim::update_particle_velocities_flip() {}
// void Sim::update_particle_velocities_pic() {
//    set_particle_velocities_from_grid();
//}
void Sim::set_particle_velocities_from_grid() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::set_particle_velocities_from_grid");
    for (size_t i = 0; i < particles.cols(); ++i) {
        particle_velocity(i) = velocity.get_vector(particle_position(i));
    }
}

void Sim::update_particles_to_samples() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_particles_to_samples");
    spdlog::info("Updating sample velocity");
    // move to the grid
    if (use_semilagrangian_fluxes) {
        particles_to_samples();
    } else {
        particles_to_samples(.1 * mesh().dx());
    }
    spdlog::info("Done updating particle velocity");
}

void Sim::update_polynomial_velocity() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_polynomial_velocity");
    const auto& S = _operator_cache.sample_to_poly_l2();

    const auto& c = sample_velocities;

    // spdlog::info("Velocity monomials: {}", velocity_stride_monomial_size());
    // spdlog::info("{}x{} + {}x{} => {}x{}", S.rows(), S.cols(), c.rows(),
    //             c.cols(), velocity.coefficients().rows(),
    //             velocity.coefficients().cols());
    velocity.coefficients() = c * S.transpose();

    polynomial_density = S * sample_density;
    // std::cout << "Velocity coeff norm: " <<  velocity.coefficients().norm()
    // << std::endl; std::cout << velocity.coefficients().transpose() <<
    // std::endl;
}
void Sim::advect_samples(double dt) {}

void Sim::advect_particles(double dt) {
    auto sw = mtao::logging::hierarchical_stopwatch("Sim::advect_particles");
    update_particle_cell_cache();
    spdlog::info("Advecting particles");
    {
        // full flip:
        // PV = PV + V - OV
        // full PIC:
        // PV = V
        // .97FLIP
        // PV = .97 * (PV + V - OV) + .03 * V
        // PV = .97 * PV - OV) + V
        // PV = .97 * PV - .97 OV + V

        // FLIP
        // std::cout << "before flip transfer: " <<
        // particle_velocities().rowwise().norm().transpose() << std::endl;
        mtao::ColVecs3d tmp = velocity.coefficients();
        // if (flip_ratio) {
        //    velocity.coefficients() =
        //        tmp - *flip_ratio * old_velocity_coefficients;
        //} else {
        //    velocity.coefficients() = tmp - old_velocity_coefficients;
        //}
        tbb::parallel_for(size_t(0), size_t(particle_cell_cache.size()),
                          [&](size_t cell_index) {
                              const auto& pidxs =
                                  particle_cell_cache[cell_index];

                              // for (auto&& [cell_index, pidxs] :
                              //     mtao::iterator::enumerate(particle_cell_cache))
                              //     {
                              for (auto&& pidx : pidxs) {
                                  auto v = velocity.get_vector(
                                      particle_position(pidx), cell_index);

                                  auto pv = particle_velocity(pidx);
                                  // if (flip_ratio) {
                                  //    pv = *flip_ratio * pv + v;
                                  //} else {
                                  //    pv += v;
                                  //}
                                  pv = v;
                              }
                          });

        velocity.coefficients() = tmp;
    }
    // std::cout << "after flip transfer: " <<
    // particle_velocities().rowwise().norm().transpose() << std::endl;
    // std::cout << "Current velocity field norm: "
    //          << velocity.coefficients().norm() << std::endl;
    // update_polynomial_velocity();
    // semilagrangian_advect_samples(dt);
    advect_particles_with_field(dt);

    update_particle_cell_cache();
}
void Sim::advect(double dt) {
    auto sw = mtao::logging::hierarchical_stopwatch("Sim::advect");
    // std::cout << "pre-advection velocities" << std::endl;
    // std::cout << "particle_velocities: " <<
    // particle_velocities().rowwise().norm().transpose() << std::endl;
    // std::cout
    // << "sample_velocities: " <<
    // sample_velocities.rowwise().norm().transpose()
    // << std::endl; std::cout << "poly velocities: " <<
    // velocity.coefficients().rowwise().norm().transpose() << std::endl;
    // update_polynomial_velocity();
    // advect_particles_with_field(dt);
    // FLIP/PIC style
    advect_particles(dt);

    if (use_semilagrangian_fluxes) {
        semilagrangian_advect_fluxes(dt);
    }

    // std::cout << "post-advection velocities" << std::endl;
    // std::cout << "particle_velocities: " <<
    // particle_velocities().rowwise().norm().transpose() << std::endl;
    // std::cout
    // << "sample_velocities: " <<
    // sample_velocities.rowwise().norm().transpose()
    // << std::endl; std::cout << "poly velocities: " <<
    // velocity.coefficients().rowwise().norm().transpose() << std::endl;
}

// void Sim::update_velocity_through_momentum_update() { int num_velocities = 0;
// }

void Sim::semilagrangian_advect_fluxes(double dt) {
    const auto& face_cob = _operator_cache.face_coboundary();
    auto B = sample_velocities.block(0, 0, 3, velocity_stride_sample_size());
    B.setZero();
    auto D = sample_density.head(velocity_stride_sample_size());
    D.setZero();
    tbb::parallel_for(int(0), mesh().face_count(), [&](int face_index) {
        auto samples = velocity_weighted_face_samples(face_index);
        auto indices =
            velocity_indexer().flux_indexer().coefficient_indices(face_index);
        if (indices.size() == 0) {
            return;
        }

        auto pblock =
            sample_velocities.block(0, *indices.begin(), 3, indices.size());
        auto dblock = sample_density.segment(*indices.begin(), indices.size());

        pblock.setZero();
        for (int j = 0; j < samples.cols(); ++j) {
            auto pt = samples.col(j).head<3>();
            auto back_pt = velocity.advect_rk2(pt, -dt);
            int cell = velocity.get_cell(back_pt);
            if (cell < 0) {
                continue;
            }
            mtao::Vec3d v = velocity.get_vector(back_pt);
            auto c = get_velocity_cell(cell);

            auto moms =
                c.evaluate_monomials_by_size(c.monomial_size(), back_pt);
            pblock += v * moms.head(pblock.cols()).transpose();
            auto cell_mom_indices =
                velocity_indexer().monomial_indexer().coefficient_indices(cell);
            auto density_mom_block = polynomial_density.segment(
                *cell_mom_indices.begin(), cell_mom_indices.size());
            dblock += density_mom_block.dot(moms) * moms.head(dblock.size());
            //    * v;
        }
        pblock /= samples.cols();
        dblock /= samples.cols();
    });
}

// void Sim::advect_point_samples_with_field(double dt) {}
// void Sim::advect_moments_with_field(double dt) {}

void Sim::advect_particles_with_field(double dt) {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::advect_particles_with_field");
    spdlog::info("Advect particles with field");
    auto vel = particle_velocities();
    // particle_positions() = velocity.advect_rk2(particle_positions(), dt);
    particle_positions() =
        velocity.advect_rk2_with_vel(particle_positions(), vel, dt);
    particle_cell_cache_dirty = true;
}
void Sim::particles_to_samples(std::optional<double> radius) {
    if (radius) {
        update_fluxes_using_particles(*radius);
        // std::cout << sample_velocities
        //                 .leftCols(velocity_flux_size())
        //                 .transpose()
        //          << std::endl;
    }
    update_moments_from_particles();
}
void Sim::update_fluxes_using_particles(double radius) {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_fluxes_using_particles");
    if (particle_cell_cache_dirty) {
        update_particle_cell_cache();
    }
    spdlog::info("Updating fluxes using particles with radius {} (grid dx = {}",
                 radius, mesh().dx());
    auto bsamps = sample_velocities.leftCols(velocity_stride_flux_size());
    bsamps.setZero();
    auto rbf =
        mtao::geometry::interpolation::SplineGaussianRadialBasisFunction<double,
                                                                         3>{};
    auto myfunc = [&](const mtao::Vec3d& a, const mtao::Vec3d& b) -> double {
        return rbf.evaluate(a, radius, b);
    };

    utils::CoefficientAccumulator3 ca(velocity_indexer());

    spdlog::info("Writing velocities");
    auto vho = ca.homogeneous_boundary_coefficients_from_point_values(
        particle_velocities().eval(), particle_positions(), particle_cell_cache,
        active_cells(), myfunc);
    bsamps = utils::dehomogenize_vector_points(vho);

    auto density_bsamps = sample_density.head(velocity_stride_flux_size());
    density_bsamps.setZero();

    spdlog::info("Writing densities");
    auto bho = ca.homogeneous_boundary_coefficients_from_point_values(
        particle_density.transpose().eval(), particle_positions(),
        particle_cell_cache, active_cells(), myfunc);
    density_bsamps = utils::dehomogenize_vector_points(bho).transpose();
}

void Sim::update_moments_from_particles() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_moments_from_particles");
    if (particle_cell_cache_dirty) {
        update_particle_cell_cache();
    }
    spdlog::info("Updating moments from particles");
    auto moments = sample_velocities.rightCols(velocity_stride_moment_size());
    moments.setZero();

    auto density_moms = sample_density.head(velocity_stride_moment_size());
    density_moms.setZero();

    tbb::parallel_for<size_t>(
        size_t(0), particle_cell_cache.size(), [&](size_t cell_index) {
            const auto& particles = particle_cell_cache[cell_index];
            // for (auto&& [cell_index, particles] :
            //     mtao::iterator::enumerate(cell_particles)) {
            auto c = get_velocity_cell(cell_index);

            if (c.moment_size() == 0) {
                return;
            }

            auto pblock = sample_velocities.block(
                0, c.global_moment_index_offset(), 3, c.moment_size());

            auto dblock = sample_density.segment(c.global_moment_index_offset(),
                                                 c.moment_size());

            pblock.setZero();
            dblock.setZero();
            auto samples = c.vertices();

            int index = 0;
            auto run = [&](auto&& pt, auto&& v, auto&& d) {
                auto block = c.evaluate_monomials_by_size(pblock.cols(), pt);
                pblock += v * block.transpose();
                dblock += d * block;
            };

            for (auto&& p : particles) {
                run(particle_position(p), particle_velocity(p),
                    particle_density(p));
            }

            int tot_size = particles.size();
            pblock /= tot_size;
            dblock /= tot_size;
        });
}
}  // namespace vem::fluidsim_3d
