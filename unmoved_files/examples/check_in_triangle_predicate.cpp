#include <Partio.h>

#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/bounding_box.hpp>
#include <nlohmann/json.hpp>
#include <vem/fluidsim_3d/sim_scene.hpp>

#include "mtao/geometry/mesh/read_obj.hpp"

//#include "vem/from_mandoline3.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options options("test", "A brief description");

    // clang-format off
    options.add_options()
        ("h,help", "Print usage");

    vem::fluidsim_3d::SimScene::add_options(options);
    // clang-format on
    options.parse_positional({"config"});
    options.positional_help({"<config>"});

    auto result = options.parse(argc, argv);
    vem::fluidsim_3d::SimScene scene(result);
    scene.create_mesh();
    // scene.active_cell_region_index = {};

    scene.create_sim(false);

    vem::fluidsim_3d::Sim& sim = *scene._sim;

    // std::cout << sim.sample_laplacian() << std::endl;
    // return 0;

    sim.static_domain = false;
    auto root = sim.inventory->real_path();
    int num_particles = sim.num_particles();
    int num_frames = 500;
    bool just_advect = false;
    if (just_advect) {
        std::vector<mtao::ColVecs3d> V;
        sim.update_polynomial_velocity();

        auto make_positions_obj = [&](int index) {
            std::ofstream ofs(root / fmt::format("positions{}.obj", index));
            if (false) {
                const auto& P = V.back();
                const auto& D = sim.particle_density;
                for (int j = 0; j < P.cols(); ++j) {
                    if (D(j) > .5) {
                        ofs << "v " << P.col(j).transpose() << std::endl;
                    }
                }
                return;
            }
            int num_frames = std::min<int>(5, V.size());
            spdlog::info("Num frames used {} with {} frames stored", num_frames,
                         V.size());
            if (num_frames == 0) {
                return;
            }
            std::vector<int> sizes;
            sizes.emplace_back(1);
            for (auto it = V.rbegin(); it < V.rbegin() + num_frames - 1; ++it) {
                sizes.emplace_back(it->cols() + sizes.back());
            }
            spdlog::info("{}", sizes);

            auto P =
                mtao::eigen::hstack_iter(V.rbegin(), V.rbegin() + num_frames);
            spdlog::info("Writing {} particles", P.cols());
            for (int j = 0; j < P.cols(); ++j) {
                ofs << "v " << P.col(j).transpose() << std::endl;
            }
            for (int k = 0; k < V.back().cols(); ++k) {
                ofs << "l ";
                // for (int j = 0; j < num_frames; ++j) {
                //    ofs << k + j * num_particles << " ";
                //}
                for (auto&& s : sizes) {
                    ofs << k + s << " ";
                }
                ofs << std::endl;
            }
        };

        bool use_mandoline = scene._mesh->type_string() == "cutmesh";
        // initialize velocities
        sim.initialize_particles(10);
        sim.initialize_mesh([](const mtao::Vec3d& p) -> mtao::Vec3d {
            return {-p.y(), p.x(), 0.0};
        });

        for (int j = 0; j < num_frames; ++j) {
            spdlog::info("Working on frame {} of {}", j, num_frames);
            auto P = sim.particle_positions();
            spdlog::info("Have {} particles", P.cols());
            V.emplace_back(std::move(P));
            make_positions_obj(j);
            sim.advect_particles_with_field(.02);
        }

    } else {
        std::vector<mtao::ColVecs3d> V;

        for (int frame_index = scene.frame_start; frame_index < scene.frame_end;
             ++frame_index) {
            auto P = sim.particle_positions();
            V.emplace_back(std::move(P));
            Partio::ParticlesDataMutable* data = Partio::create();
            Partio::ParticleAttribute pos_attr = data->addAttribute(
                "position", Partio::ParticleAttributeType::VECTOR, 3);

            Partio::ParticleAttribute vel_attr = data->addAttribute(
                "velocity", Partio::ParticleAttributeType::VECTOR, 3);

            Partio::ParticleAttribute id_attr =
                data->addAttribute("id", Partio::ParticleAttributeType::INT, 1);

            int count = (sim.particle_density.array() > .5).count();
            if (count > 0) {
                auto it = data->addParticles(count);
                for (int j = 0; j < sim.num_particles(); ++j) {
                    double d = sim.particle_density(j);
                    if (d > .5) {
                        auto p = sim.particle_position(j);
                        auto v = sim.particle_velocity(j);
                        float* dat = data->dataWrite<float>(pos_attr, it.index);
                        dat[0] = p.x();
                        dat[1] = p.y();
                        dat[2] = p.z();
                        dat = data->dataWrite<float>(vel_attr, it.index);
                        dat[0] = v.x();
                        dat[1] = v.y();
                        dat[2] = v.z();
                        data->dataWrite<int>(vel_attr, it.index)[0] = j;
                        it++;
                    }
                }
            }
            Partio::write(
                (root / fmt::format("output-{:04d}.bgeo", frame_index)).c_str(),
                *data);
            scene.step(frame_index);
        }
    }
    return 0;
}
