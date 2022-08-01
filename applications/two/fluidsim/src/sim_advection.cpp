#include <tbb/parallel_for.h>

#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/two/boundary_facets.hpp>
#include <vem/two/cells_adjacent_to_edge.hpp>
#include <vem/utils/dehomogenize_vector_points.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "vem/two/fluidsim/sim.hpp"
#include "vem/two/coefficient_accumulator.hpp"
namespace vem::two::fluidsim {
// void Sim::update_particle_velocities_flip() {}
// void Sim::update_particle_velocities_pic() {
//    set_particle_velocities_from_grid();
//}
void Sim::set_particle_velocities_from_grid() {
    for (size_t i = 0; i < particles.cols(); ++i) {
        particle_velocity(i) = velocity.get_vector(particle_position(i));
    }
}

void Sim::update_sample_data(double dt) {
    // move to the grid
    if (use_semilagrangian_fluxes) {
        spdlog::info("Updating velocities using semilagrangian advectino");
        particle_to_sample(dt);
    } else {
        spdlog::info("Updating rbfs");
        particle_to_sample(dt, mesh().dx() / 5);
    }
}

void Sim::update_polynomial_velocity() {
    auto S = sample_to_poly_l2();

    const auto &c = sample_velocities;

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
    update_particle_cell_cache();
    {
        // weighed PIC + FLIP
        // v' = alpha * ( v + new_vel - old_vel) + ( 1 - alpha ) * new_vel
        // v' = new_vel + (alpha) ( v - old_vel)
        // v' = (new_vel - alpha old_vel) - alpha * v
        // FLIP
        // std::cout << "before flip transfer: " <<
        // particle_velocities().rowwise().norm().transpose() << std::endl;
        // mtao::ColVecs2d tmp = velocity.coefficients();
        // velocity.coefficients() = tmp - flip_ratio *
        // old_velocity_coefficients;
        // std::cout << "temporary diff thing: "
        //          << velocity.coefficients().rowwise().norm().transpose()
        //          << std::endl;
        const auto &cvelocity = velocity;
        tbb::parallel_for<int>(
            0, particle_cell_cache.size(), [&](int cell_index) {
                const auto &pidxs = particle_cell_cache[cell_index];
                for (const auto &pidx : pidxs) {
                    const mtao::Vec2d v = cvelocity.get_vector(
                        particle_position(pidx), cell_index);
                    auto vel = particle_velocity(pidx);
                    // vel = v + flip_ratio * vel;
                    vel = v;
                }
            });

        // velocity.coefficients() = tmp;
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

mtao::ColVecs2d Sim::advect_points_rk2(const mtao::ColVecs2d &P, double dt) {
    return velocity.advect_rk2(P, dt);
}
auto Sim::semilagrangian_advected_edge_sample_velocities(double dt)
    -> mtao::ColVecs2d {
    auto edge_cob = edge_coboundary_map(mesh(), active_cells());
    auto edge_cell_neighbors =
        cells_adjacent_to_edge(mesh(), active_cells());
    mtao::ColVecs3d R(2, velocity_stride_sample_size());
    R.setZero();

    return coefficients_from_point_sample_vector_function(
        [&](const mtao::Vec2d &pt) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
            /*
                  auto back_pt = velocity.advect_rk2(pt, -dt);
                  int cell = velocity.get_cell(back_pt);
                  if (cell < 0) {
                      return mtao::Vec2d::Zero();
                  }
                  return velocity.get_vector(back_pt);
              */
        }

    );
}
#if defined(VEM_FLUX_MOMENT_FLUID)

void Sim::semilagrangian_advect_fluxes(double dt) {
    auto edge_cob = edge_coboundary_map(mesh(), active_cells());
    auto edge_cell_neighbors =
        cells_adjacent_to_edge(mesh(), active_cells());
    auto B = sample_velocities.block(0, 0, 2, velocity_stride_sample_size());
    B.setZero();
    auto D = sample_density.head(velocity_stride_sample_size());
    D.setZero();
    tbb::parallel_for(int(0), mesh().edge_count(), [&](int edge_index) {
        auto samples = velocity_weighted_edge_samples(edge_index);
        auto indices =
            velocity_indexer().flux_indexer().coefficient_indices(edge_index);
        if (indices.size() == 0) {
            return;
        }

        auto pblock =
            sample_velocities.block(0, *indices.begin(), 2, indices.size());
        auto dblock = sample_density.segment(*indices.begin(), indices.size());

        pblock.setZero();
        for (int j = 0; j < samples.cols(); ++j) {
            auto pt = samples.col(j).head<2>();
            auto back_pt = velocity.advect_rk2(pt, -dt);
            int cell = velocity.get_cell(back_pt);
            if (cell < 0) {
                continue;
            }
            mtao::Vec2d v = velocity.get_vector(back_pt);
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
#else
void Sim::semilagrangian_advect_samples(double dt) {
    auto P = velocity_indexer().point_sample_indexer().get_positions();
}
#endif
// void Sim::advect_point_samples_with_field(double dt) {}
// void Sim::advect_moments_with_field(double dt) {}

void Sim::advect_particles_with_field(double dt) {
    tbb::parallel_for(int(0), int(particles.cols()), [&](int pidx) {
        // rk2
        auto p = particle_position(pidx);
        p = velocity.advect_rk2(p, dt);
        // if ((old - p).norm() == 0) {
        //    spdlog::info(" Particle did not move");
        //}
        // std::cout << "Particle moved from " << old.transpose() << " => "
        //          << p.transpose() << std::endl;
    });
    particle_cell_cache_dirty = true;
}
void Sim::particle_to_sample(double dt, std::optional<double> radius) {
    if (radius) {
        spdlog::info("Updating particle velocities using particle radius");
#if defined(VEM_FLUX_MOMENT_FLUID)
        update_fluxes_using_particles(*radius);
        // update_fluxes_using_semilag_and_particles(dt, *radius);
#else
        update_point_samples_using_particles(*radius);
#endif
        // std::cout << sample_velocities
        //                 .leftCols(velocity_stride_point_sample_size())
        //                 .transpose()
        //          << std::endl;
    }
    update_moments_from_particles();
}
#if defined(VEM_FLUX_MOMENT_FLUID)
void Sim::update_fluxes_using_particles(double radius) {
    if (particle_cell_cache_dirty) {
        update_particle_cell_cache();
    }

    auto bsamps = sample_velocities.leftCols(velocity_stride_flux_size());
    spdlog::info("Velocity sample sizes: {} / {}", velocity_stride_flux_size(),
                 sample_velocities.cols());
    bsamps.setZero();
    auto rbf =
        mtao::geometry::interpolation::SplineGaussianRadialBasisFunction<double,
                                                                         2>{};
    auto myfunc = [&](const mtao::Vec2d &a, const mtao::Vec2d &b) -> double {
        return rbf.evaluate(a, radius, b);
    };

    CoefficientAccumulator<FluxMomentIndexer> ca(velocity_indexer());
    auto vho = ca.homogeneous_boundary_coefficients_from_point_values(
        particle_velocities().eval(), particle_positions(), particle_cell_cache,
        active_cells(), myfunc);
    bsamps = utils::dehomogenize_vector_points(vho);

    auto density_bsamps = sample_density.head(velocity_stride_flux_size());
    density_bsamps.setZero();

    auto bho = ca.homogeneous_boundary_coefficients_from_point_values(
        particle_density.transpose().eval(), particle_positions(),
        particle_cell_cache, active_cells(), myfunc);
    density_bsamps = utils::dehomogenize_vector_points(bho).transpose();

    /*
    auto edge_cob = utils::edge_coboundary_map(mesh(), active_cells());
    auto edge_cell_neighbors =
        utils::cells_adjacent_to_edge(mesh(), active_cells());
    auto rbf =
        mtao::geometry::interpolation::SplineGaussianRadialBasisFunction<double,
                                                                         2>{};
    mtao::VecXd weight_sums(velocity_indexer().flux_size());
    weight_sums.setZero();
    auto B = sample_velocities.block(0, 0, 2,
velocity_stride_sample_size()); B.setZero(); auto D =
sample_density.head(velocity_stride_sample_size()); D.setZero(); int
edge_index = 0; #pragma omp parallel for for (edge_index = 0; edge_index
< mesh().edge_count(); ++edge_index) { auto samples =
velocity_weighted_edge_samples(edge_index); const auto &cob =
edge_cob.at(edge_index); const auto &cell_neighbors =
edge_cell_neighbors.at(edge_index); int sample_count = cob.size() *
samples.cols(); auto indices =
            velocity_indexer().flux_indexer().coefficient_indices(edge_index);
        if (indices.size() == 0) {
            continue;
        }

        auto pblock =
            sample_velocities.block(0, *indices.begin(), 2,
indices.size()); auto dblock = sample_density.segment(*indices.begin(),
indices.size()); auto wblock = weight_sums.segment(*indices.begin(),
indices.size());

        for (auto &&cell_index : cell_neighbors) {
            const auto &particles = particle_cell_cache[cell_index];
            auto c = get_velocity_cell(cell_index);
            for (auto &&p : particles) {
                auto pposition = particle_position(p);
                auto pvelocity = particle_velocity(p);
                auto pdensity = particle_density(p);
                // std::cout << pvelocity.transpose() << std::endl;

                for (int j = 0; j < samples.cols(); ++j) {
                    auto pt = samples.col(j).head<2>();

                    double weight =
                        rbf.evaluate(pposition, radius,
                                     pt);  // * (p -
center).normalized();

                    auto moms =
c.evaluate_monomials_by_size(pblock.cols(), pt); pblock += weight *
pvelocity * moms.transpose(); dblock += weight * pdensity *
moms.transpose(); wblock.array() += weight;
                    //    * v;
                }
            }
        }
    }
    weight_sums.array() = (weight_sums.array().abs() > 1e-5)
                              .select(1.0 / weight_sums.array(), 0.0);
    B = B * weight_sums.asDiagonal();
    D = weight_sums.asDiagonal() * D;
    */
}
void Sim::update_fluxes_using_semilag_and_particles(double dt, double radius) {
    if (particle_cell_cache_dirty) {
        update_particle_cell_cache();
    }

    auto rbf =
        mtao::geometry::interpolation::SplineGaussianRadialBasisFunction<double,
                                                                         2>{};
    /*
    auto edge_cob = utils::edge_coboundary_map(mesh(), active_cells());
    auto edge_cell_neighbors =
        utils::cells_adjacent_to_edge(mesh(), active_cells());

    // auto active_edge = [&](int edge_index) -> bool {
    //    if(edge_index < edge_cob.size() || >= edge_cob.size()) {
    //        return false;
    //    } else {
    //        const auto& cob = edge_cob[edge_index];
    //        return cob.size() == 2;
    //};
    mtao::VecXd weight_sums(velocity_indexer().flux_size());
    weight_sums.setZero();
    auto B = sample_velocities.block(0, 0, 2,
velocity_stride_sample_size()); B.setZero(); auto D =
sample_density.head(velocity_stride_sample_size()); D.setZero(); int
edge_index = 0;
    // mtao::vector<mtao::Vec4d> backprojections;
#pragma omp parallel for
    for (edge_index = 0; edge_index < mesh().edge_count(); ++edge_index) {
        auto samples = velocity_weighted_edge_samples(edge_index);
        const auto &cob = edge_cob.at(edge_index);
        const auto &cell_neighbors = edge_cell_neighbors.at(edge_index);
        int sample_count = cob.size() * samples.cols();
        auto indices =
            velocity_indexer().flux_indexer().coefficient_indices(edge_index);
        if (indices.size() == 0) {
            continue;
        }
        const bool is_interior = cob.size() == 2;

        auto pblock =
            sample_velocities.block(0, *indices.begin(), 2, indices.size());
        auto dblock = sample_density.segment(*indices.begin(),
indices.size()); auto wblock = weight_sums.segment(*indices.begin(),
indices.size());

        if (is_interior) {
            for (int j = 0; j < samples.cols(); ++j) {
                auto pt = samples.col(j).head<2>();
                auto back_pt = velocity.advect_rk2(pt, -dt);
                mtao::Vec4d phase = mtao::eigen::vstack(back_pt, pt -
back_pt); phase = mtao::eigen::vstack(pt, velocity.get_vector(pt));

                // backprojections.emplace_back(phase);
                int cell = velocity.get_cell(back_pt);
                if (!cob.contains(cell)) {
                    continue;
                }

                mtao::Vec2d pvelocity = velocity.get_vector(back_pt);
                auto c = get_velocity_cell(cell);

                // auto [start, end] =
                // c.monomial_indexer().coefficient_range(cell); int
                // num_densities = end - start; auto density_block =
                // polynomial_density.segment(start, num_densities);
                auto moms = c.evaluate_monomials_by_size(pblock.cols(), pt);
                pblock += pvelocity * moms.transpose();
                // dblock += density_block.dot(moms) * moms.transpose();
                wblock.array() += 1.0;
                //    * v;
            }
        } else {
            // wblock.array() += samples.cols();
        }

        for (auto &&cell_index : cell_neighbors) {
            const auto &particles = particle_cell_cache[cell_index];
            auto c = get_velocity_cell(cell_index);
            for (auto &&p : particles) {
                auto pposition = particle_position(p);
                auto pvelocity = particle_velocity(p);
                auto pdensity = particle_density(p);
                // std::cout << pvelocity.transpose() << std::endl;

                for (int j = 0; j < samples.cols(); ++j) {
                    auto pt = samples.col(j).head<2>();

                    double weight =
                        rbf.evaluate(pposition, radius,
                                     pt);  // * (p - center).normalized();

                    auto moms = c.evaluate_monomials_by_size(pblock.cols(),
pt); pblock += weight * pvelocity * moms.transpose(); dblock += weight *
pdensity * moms.transpose(); wblock.array() += weight;
                    //    * v;
                }
            }
        }
    }
    weight_sums.array() = (weight_sums.array().abs() > 1e-5)
                              .select(1.0 / weight_sums.array(), 0.0);
    B = B * weight_sums.asDiagonal();
    D = weight_sums.asDiagonal() * D;
    // if (active_inventory) {
    //    serialization::serialize_points4(
    //        *active_inventory, "backpropagation_directions",
    //        mtao::eigen::stl2eigen(backprojections));
    //    auto& meta =
    // active_inventory->asset_metadata("backpropagation_directions");
    //    meta["type"] = "point2,velocity2";
    //}
    */
}
#else
void Sim::update_point_samples_using_particles(double radius) {
    if (particle_cell_cache_dirty) {
        update_particle_cell_cache();
    }

    auto rbf =
        mtao::geometry::interpolation::SplineGaussianRadialBasisFunction<double,
                                                                         2>{};

    auto f = [&](auto &&v) {
        return rbf.evaluate(
            s, radius,
            particle_position(p));  // * (p - center).normalized();
    };
    mtao::VecXd weight_sums(velocity_indexer().point_sample_size());
    weight_sums.setZero();
    auto B = sample_velocities.leftCols(weight_sums.size());
    B.setZero();
    auto D = sample_density.head(weight_sums.size());
    D.setZero();
    for (auto &&[cell_index, particles] :
         mtao::iterator::enumerate(particle_cell_cache)) {
        auto c = get_velocity_cell(cell_index);
        for (auto &&point_sample_index : c.point_sample_indices()) {
            auto s = velocity_indexer().point_sample_indexer().get_position(
                point_sample_index);
            auto sample_vel = sample_velocities.col(point_sample_index);
            auto &sample_den = sample_density(point_sample_index);
            double &weight_sum = weight_sums(point_sample_index);
            for (auto &&p : particles) {
                double weight = rbf.evaluate(
                    s, radius,
                    particle_position(p));  // * (p - center).normalized();

                weight_sum += weight;
                sample_vel += weight * particle_velocity(p);
                sample_den += weight * particle_density(p);
            }
        }
    }
    weight_sums.array() = (weight_sums.array().abs() > 1e-5)
                              .select(1.0 / weight_sums.array(), 0.0);
    B = B * weight_sums.asDiagonal();
    D = weight_sums.asDiagonal() * D;
}
#endif

void Sim::update_moments_from_particles() {
    if (particle_cell_cache_dirty) {
        update_particle_cell_cache();
    }
    auto moments = sample_velocities.rightCols(velocity_stride_moment_size());

    CoefficientAccumulator<FluxMomentIndexer> ca(velocity_indexer());
    auto homo_coeffs = ca.homogeneous_interior_coefficients_from_point_values(
        particle_velocities().eval(), particle_positions(), particle_cell_cache,
        active_cells());
    moments = utils::dehomogenize_vector_points(homo_coeffs);
    std::cout << homo_coeffs << std::endl;

    if (do_buoyancy) {
        auto density_moms = sample_density.tail(velocity_stride_moment_size());
        density_moms.setZero();
        spdlog::info("Particle density sum: {}", particle_density.sum());

        auto homo_coeffs =
            ca.homogeneous_interior_coefficients_from_point_values(
                particle_density.transpose().eval(), particle_positions(),
                particle_cell_cache, active_cells());
        density_moms =
            utils::dehomogenize_vector_points(homo_coeffs).transpose();
    }
}
}  // namespace vem::fluidsim_2d
