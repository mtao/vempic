#include <omp.h>

#include <fstream>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/types.hpp>
#include <vem/creator2.hpp>
#include <vem/flux_moment_indexer.hpp>
#include <vem/point_moment_indexer.hpp>
#include <vem/serialization/frame_inventory.hpp>
#include <vem/serialization/inventory.hpp>
#include <vem/serialization/serialize_eigen.hpp>

#include "vem/cell.hpp"

using namespace vem;

auto inventory =
    serialization::Inventory::from_scratch("moment_reconstruction_experiment");
std::optional<serialization::FrameInventory> frame_inventory;

std::array<mtao::VecXd, 2> point_moment_reconstructions(
    const vem::VEMMesh2& mesh,
    const std::function<double(const mtao::Vec2d&)>& f, int max_degree,
    const mtao::ColVecs2d& P,
    const std::vector<std::set<int>>& cell_particles) {
    spdlog::warn("Point moment recon");
    PointMomentIndexer pmi(mesh, max_degree);
    auto C =
        pmi.coefficients_from_point_sample_function(f, P, cell_particles, {});
    // std::cout << "Cell data: " << C.transpose() << std::endl;
    // std::cout << "L2\n"
    //          << Eigen::MatrixXd(pmi.sample_to_poly_l2()) << std::endl;
    // std::cout << "Dirichlet\n"
    //          << Eigen::MatrixXd(pmi.sample_to_poly_dirichlet()) << std::endl;
    // std::cout << std::endl;
    mtao::VecXd L2P = pmi.sample_to_poly_l2() * C;
    mtao::VecXd DP = pmi.sample_to_poly_dirichlet() * C;

    serialization::serialize_points4(*frame_inventory, "point_moment_l2_field",
                                     L2P);
    frame_inventory->asset_metadata("point_moment_l2_field")["type"] =
        "point2,velocity2";

    serialization::serialize_points4(*frame_inventory,
                                     "point_moment_dirichlet_field", DP);
    frame_inventory->asset_metadata("point_moment_dirichlet_field")["type"] =
        "point2,velocity2";
    return {{L2P, DP}};
}

std::array<mtao::VecXd, 2> flux_moment_reconstructions(
    const vem::VEMMesh2& mesh,
    const std::function<double(const mtao::Vec2d&)>& f, int max_degree,
    const mtao::ColVecs2d& P,
    const std::vector<std::set<int>>& cell_particles) {
    spdlog::warn("Flux moment recon");
    FluxMomentIndexer fmi(mesh, max_degree);
    auto C =
        fmi.coefficients_from_point_sample_function(f, P, cell_particles, {});

    // std::cout << "Cell data: " << C.transpose() << std::endl;
    // std::cout << "L2\n"
    //          << Eigen::MatrixXd(fmi.sample_to_poly_l2()) << std::endl;
    // std::cout << "Dirichlet\n"
    //          << Eigen::MatrixXd(fmi.sample_to_poly_dirichlet()) << std::endl;
    // std::cout << std::endl;

    mtao::VecXd L2P = fmi.sample_to_poly_l2() * C;
    mtao::VecXd DP = fmi.sample_to_poly_dirichlet() * C;

    serialization::serialize_points4(*frame_inventory, "flux_moment_l2_field",
                                     L2P);
    frame_inventory->asset_metadata("flux_moment_l2_field")["type"] =
        "point2,velocity2";

    serialization::serialize_points4(*frame_inventory,
                                     "flux_moment_dirichlet_field", DP);
    frame_inventory->asset_metadata("flux_moment_dirichlet_field")["type"] =
        "point2,velocity2";
    return {{L2P, DP}};
}

mtao::VecXd least_squares_reconstruction(
    const vem::VEMMesh2& mesh,
    const std::function<double(const mtao::Vec2d&)>& f, int max_degree,
    const mtao::ColVecs2d& P,
    const std::vector<std::set<int>>& cell_particles) {
    vem::MonomialBasisIndexer mbi(mesh, max_degree);
    mtao::VecXd A(mbi.num_coefficients());

    tbb::parallel_for(int(0), cell_particles.size(), [&](int cell_index) {
        auto& particles = cell_particles[cell_index];
        // for (auto&& [cell_index, particles] :
        //     mtao::iterator::enumerate(cell_particles)) {
        VEM2Cell c(mesh, cell_index);

        auto [start, end] = mbi.coefficient_range(cell_index);
        auto block = A.segment(start, end - start);

        mtao::ColVecs2d LP(2, particles.size());
        mtao::VecXd V(particles.size());
        int index = 0;
        for (auto&& p : particles) {
            auto pt = P.col(p);
            double v = f(pt);

            LP.col(index) = pt;
            V(index) = v;
            index++;
        }

        auto LQFit = c.unweighted_least_squares_coefficients(max_degree, LP, V);

        block = LQFit;
    });

    // std::cout << "Least squares coeffs: " << A.transpose() << std::endl;
    FluxMomentIndexer fmi(mesh, max_degree);
    auto ME = fmi.get_cell(0).monomial_evaluation();
    // std::cout << "ME size: " << ME.rows() << " " << ME.cols() << std::endl;
    // std::cout << "ME\n" << ME << std::endl;
    // std::cout << "Monomial eval: " << (ME * A).transpose() << std::endl;
    serialization::serialize_points4(*frame_inventory, "least_squares_field",
                                     A);
    frame_inventory->asset_metadata("least_squares_field")["type"] =
        "point2,velocity2";

    return A;
}

void run(const vem::VEMMesh2& mesh,
         const std::function<double(const mtao::Vec2d&)>& f, int max_degree,
         int frame_index) {
    int sample_count = mesh.cell_count() * frame_index;
    mtao::ColVecs2d P(2, sample_count);
    std::vector<std::set<int>> particle_ownerships(mesh.cell_count());

    srand(0);
    auto bbox = mesh.bounding_box();
    double bbvol = bbox.sizes().prod();
    double per_cell = bbvol / mesh.cell_count() / (1 + frame_index);
    double rad = std::sqrt(per_cell / (M_PI)) / 5;
    P = mtao::geometry::point_cloud::bridson_poisson_disk_sampling(
        mesh.bounding_box(), rad);

    for (int j = 0; j < P.cols(); ++j) {
        auto p = P.col(j);
        int cell = mesh.get_cell(P.col(j));

        particle_ownerships[cell].emplace(j);
    }
    spdlog::info(
        "Instantiating bridson poisson disk with radius {}, got {} elements",
        rad, P.cols());

    serialization::serialize_points2(*frame_inventory, "samples", P);
    frame_inventory->asset_metadata("samples")["type"] = "point2";

    auto LSR = least_squares_reconstruction(mesh, f, max_degree, P,
                                            particle_ownerships);
    auto [FML2, FMD] = flux_moment_reconstructions(mesh, f, max_degree, P,
                                                   particle_ownerships);
    auto [PML2, PMD] = point_moment_reconstructions(mesh, f, max_degree, P,
                                                    particle_ownerships);

    frame_inventory = {};
}

int main(int argc, char* argv[]) {
    nlohmann::json js;

    int max_degree = std::stoi(argv[1]);
    int desired_samples_per_cell = std::stoi(argv[2]);
    VEMMesh2Creator creator;
    if (argc > 3) {
        std::ifstream(argv[3]) >> js;
        creator.configure_from_json(js, true);
    }

    if (!creator.stored_mesh()) {
        creator.make_grid_mesh();
    }

    inventory.set_immediate_mode();
    inventory.add_metadata("degree", max_degree);
    inventory.add_metadata(
        "visualization_manifest",
        nlohmann::json::object(
            {{"point_moment_dirichlet_field", "point2,velocity2"},
             {"point_moment_l2_field", "point2,velocity2"},
             {"flux_moment_dirichlet_field", "point2,velocity2"},
             {"flux_moment_l2_field", "point2,velocity2"},
             {"least_squares_field", "point2,velocity2"},
             {"point_moment_dirichlet_points", "point2,velocity2"},
             {"point_moment_l2_points", "point2,velocity2"},
             {"flux_moment_dirichlet_points", "point2,velocity2"},
             {"flux_moment_l2_points", "point2,velocity2"},
             {"least_squares_points", "point2,velocity2"},
             {"samples", "point2"}}));

    std::ofstream(inventory.get_new_asset_path("mesh_info"))
        << creator.serialize_to_json();
    auto m = creator.stored_mesh();
    for (int j = 0; j < desired_samples_per_cell; ++j) {
        frame_inventory =
            serialization::FrameInventory::for_creation(inventory, j);
        auto f = [](const mtao::Vec2d& f) -> double {
            return std::sin(1.0 / ((f.array() - .5).matrix().norm() + 1e-5));
            return f.y();
            return f.x() * f.x();
        };
        run(*m, f, max_degree, j);
        // auto [A, B] = reconstruct_from_function(*m, f, max_degree, j);
        // std::cout << mtao::eigen::hstack(A, B).transpose() << " " <<
        // std::endl; std::cout << std::endl;
        frame_inventory.reset();
    }
}

