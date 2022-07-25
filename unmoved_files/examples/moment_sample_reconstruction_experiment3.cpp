#include <omp.h>

//#include <tbb/task_scheduler_init.h>
#include <cxxopts.hpp>
#include <fstream>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/types.hpp>
#include <vem/creator3.hpp>
#include <vem/flux_moment_indexer3.hpp>
#include <vem/serialization/frame_inventory.hpp>
#include <vem/serialization/inventory.hpp>
#include <vem/serialization/serialize_eigen.hpp>

#include "vem/cell.hpp"

using namespace vem;

auto inventory =
    serialization::Inventory::from_scratch("moment_reconstruction_experiment");
std::optional<serialization::FrameInventory> frame_inventory;

std::optional<int> active_region_index;
std::set<int> active_cells;
template <typename Derived, typename VDerived>
double evaluate_single_polynomial(const Eigen::MatrixBase<Derived>& p,
                                  const MonomialBasisIndexer3& indexer, int cell,
                                  const Eigen::MatrixBase<VDerived>& coeffs) {
    auto [start, end] = indexer.coefficient_range(cell);
    int num_coeffs = end - start;
    auto coeff_block = coeffs.segment(start, num_coeffs);
    double ret = 0;
    for (int k = 0; k < num_coeffs; ++k) {
        ret += coeff_block(k) * indexer.evaluate_monomial(cell, k, p);
    }
    return ret;
}

template <typename Derived, typename VDerived>
mtao::VecXd evaluate_polynomials(const Eigen::MatrixBase<Derived>& P,
                                 const MonomialBasisIndexer3& indexer,
                                 const Eigen::MatrixBase<VDerived>& coeffs) {
    mtao::VecXd ret(P.cols());
    // auto cells = indexer.mesh().get_cells(P);
    for (int j = 0; j < P.cols(); ++j) {
        auto p = P.col(j);
        // int cell = cells(j);
        int cell = indexer.mesh().get_cell(p);
        auto [start, end] = indexer.coefficient_range(cell);

        int num_coeffs = end - start;
        auto coeff_block = coeffs.segment(start, num_coeffs);
        ret(j) = 0;
        for (int k = 0; k < num_coeffs; ++k) {
            ret(j) += coeff_block(k) * indexer.evaluate_monomial(cell, k, p);
        }
    }
    return ret;
}

template <typename Indexer>
void add_samples(const mtao::ColVecs3d& P, const Indexer& indexer,
                 const mtao::VecXd& coefficients, const std::string& prefix) {
    mtao::RowVecXd D =
        evaluate_polynomials(P, indexer.monomial_indexer(), coefficients)
            .transpose();

    {
        std::string name = prefix + "_reconstruction";
        mtao::ColVecs4d V = mtao::eigen::vstack(P, D);
        serialization::serialize_points4(*frame_inventory, name, V);
        frame_inventory->asset_metadata(name)["type"] = "point3,density1";
    }
}


std::array<mtao::VecXd, 2> flux_moment_reconstructions(
    const vem::VEMMesh3& mesh,
    const std::function<double(const mtao::Vec3d&)>& f, int max_degree,
    const mtao::ColVecs3d& P,
    const std::vector<std::set<int>>& cell_particles) {
    spdlog::warn("Flux moment recon");
    FluxMomentIndexer3 fmi(mesh, max_degree);
    auto C =
        fmi.coefficients_from_point_sample_function(f, P, cell_particles, {});

    std::cout << "Coeffs from point sample func:\n";
    std::cout << C.transpose() << std::endl;

    auto c = fmi.get_cell(0);
    std::cout << "Mono eval\n" << c.monomial_evaluation() << std::endl;

    mtao::VecXd L2P = fmi.sample_to_poly_l2(active_cells) * C;
    mtao::VecXd DP = fmi.sample_to_poly_dirichlet(active_cells) * C;

    serialization::serialize_VecXd(*frame_inventory, "flux_moment_l2_field",
                                   L2P);
    frame_inventory->asset_metadata("flux_moment_l2_field")["type"] =
        "scalar_field";

    serialization::serialize_VecXd(*frame_inventory,
                                   "flux_moment_dirichlet_field", DP);
    frame_inventory->asset_metadata("flux_moment_dirichlet_field")["type"] =
        "scalar_field";
    add_samples(P, fmi, L2P, "flux_moment_l2");
    add_samples(P, fmi, DP, "flux_moment_dirichlet");
    return {{L2P, DP}};
}

mtao::VecXd least_squares_reconstruction(
    const vem::VEMMesh3& mesh,
    const std::function<double(const mtao::Vec3d&)>& f, int max_degree,
    const mtao::ColVecs3d& P,
    const std::vector<std::set<int>>& cell_particles) {
    vem::MonomialBasisIndexer3 mbi(mesh, max_degree);
    mtao::VecXd A(mbi.num_coefficients());

    int cell_index = 0;
#pragma omp parallel for
    for (cell_index = 0; cell_index < cell_particles.size(); ++cell_index) {
        auto& particles = cell_particles[cell_index];
        // for (auto&& [cell_index, particles] :
        //     mtao::iterator::enumerate(cell_particles)) {
        VEM3Cell c(mesh, cell_index);

        auto [start, end] = mbi.coefficient_range(cell_index);
        auto block = A.segment(start, end - start);
        if (!active_cells.contains(cell_index)) {
            block.setConstant(0);
            continue;
        }

        mtao::ColVecs3d LP(3, particles.size());
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
    }

    // std::cout << "Least squares coeffs: " << A.transpose() << std::endl;
    FluxMomentIndexer3 fmi(mesh, max_degree);
    auto ME = fmi.get_cell(0).monomial_evaluation();
    // std::cout << "ME size: " << ME.rows() << " " << ME.cols() << std::endl;
    // std::cout << "ME\n" << ME << std::endl;
    // std::cout << "Monomial eval: " << (ME * A).transpose() << std::endl;
    serialization::serialize_VecXd(*frame_inventory, "least_squares_field", A);
    frame_inventory->asset_metadata("least_squares_field")["type"] =
        "scalar_field";

    add_samples(P, fmi, A, "least_squares");
    return A;
}

void run(const vem::VEMMesh3& mesh,
         const std::function<double(const mtao::Vec3d&)>& f, int max_degree,
         int frame_index) {
    //int sample_count = mesh.cell_count() * (1 + frame_index);
    mtao::ColVecs3d P;//(3, sample_count);
    std::vector<std::set<int>> particle_ownerships(mesh.cell_count());

    srand(0);
    auto bbox = mesh.bounding_box();
    double bbvol = bbox.sizes().prod();
    double per_cell = bbvol / mesh.cell_count() / (1 + frame_index);
    spdlog::info("Mesh dx: {}", mesh.dx());
    double rad = mesh.dx() / (1 + frame_index);
    spdlog::info("Time for bridson: {}", rad);
    P = mtao::geometry::point_cloud::bridson_poisson_disk_sampling(
        mesh.bounding_box(), rad);

    mtao::RowVecXd D(P.cols());
    for (int j = 0; j < P.cols(); ++j) {
        auto p = P.col(j);
        int cell = mesh.get_cell(P.col(j));

        particle_ownerships[cell].emplace(j);
        D(j) = f(p);
    }
    spdlog::info(
        "Instantiating bridson poisson disk with radius {}, got {} elements",
        rad, P.cols());

    mtao::ColVecs4d V = mtao::eigen::vstack(P, D);
    serialization::serialize_points4(*frame_inventory, "samples", V);
    frame_inventory->asset_metadata("samples")["type"] = "point3,density1";

    auto LSR = least_squares_reconstruction(mesh, f, max_degree, P,
                                            particle_ownerships);
    auto [FML2, FMD] = flux_moment_reconstructions(mesh, f, max_degree, P,
                                                   particle_ownerships);

    FluxMomentIndexer3 fmi(mesh, max_degree);

    mtao::RowVecXd LSRC =
        evaluate_polynomials(P,fmi.monomial_indexer(), LSR)
            .transpose();

    mtao::RowVecXd FMDC =
        evaluate_polynomials(P,fmi.monomial_indexer(), FMD)
            .transpose();
    mtao::RowVecXd FML2C =
        evaluate_polynomials(P,fmi.monomial_indexer(), FML2)
            .transpose();

    spdlog::info("LSRC: {}", (LSRC-D).lpNorm<Eigen::Infinity>());
    std::cout << LSR.transpose() << std::endl;
    spdlog::info("LML2C: {}", (FML2C-D).lpNorm<Eigen::Infinity>());
    std::cout << FML2.transpose() << std::endl;
    spdlog::info("LMDC: {}", (FMDC-D).lpNorm<Eigen::Infinity>());
    std::cout << FMD.transpose() << std::endl;
    

    //auto write_error = [&](const std::string& prefix) {
    //    const std::string recon = prefix + "_reconstruction";
    //    auto PR = serialization::deserialize_points4(*frame_inventory, recon);
    //    spdlog::info("{} error: {}",prefix, (PR.row(3) - D).lpNorm<Eigen::Infinity>());
    //};

    //write_error("flux_moment_l2");
    //write_error("flux_moment_dirichlet");
    //write_error("least_squares");
    //frame_inventory = {};
}

int main(int argc, char* argv[]) {
    //tbb::task_scheduler_init init(1);
    nlohmann::json js;

    cxxopts::Options options("test", "A brief description");

    // clang-format off
    options.add_options()
        ("d,degree", "polynomial degree of VEM used", cxxopts::value<int>()->default_value("1"))
        ("s,samples", "number of samples in each cell", cxxopts::value<int>()->default_value("5"))
        ("config", "configuration json file", cxxopts::value<std::string>())
        ("l,linear", "use a linear function")
        ("q,quadratic", "use a quadratic function")
        ("x,quad_x", "x coord of quad center", cxxopts::value<double>()->default_value("0.0"))
        ("y,quad_y", "y coord of quad center", cxxopts::value<double>()->default_value("0.0"))
        ("a,angle", "the angle of the gradient of the linear function in degrees", cxxopts::value<double>()->default_value("0.0"))
        ("h,help", "Print usage");
    // clang-format on
    options.parse_positional({"config"});
    options.positional_help({"<mesh_file> <output_grid>"});

    auto result = options.parse(argc, argv);

    bool help_out = result.count("help");
    if (help_out) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    int max_degree = result["degree"].as<int>();
    int desired_samples_per_cell = result["samples"].as<int>();
    VEMMesh3Creator creator;

    bool use_quad = result.count("quadratic");
    bool use_linear = result.count("linear");
    if (result.count("config") == 0) {
        creator.grid_mesh_dimensions = {{2,2,2}};
        creator.grid_mesh_bbox.min().setConstant(-1);
        creator.grid_mesh_bbox.max().setConstant(1);
        creator.make_grid_mesh();
    } else {
        std::ifstream(result["config"].as<std::string>()) >> js;
        creator.configure_from_json(js, true);
    }

    if (!creator.stored_mesh()) {
        spdlog::error("Was unable to create a mesh");
        return 1;
    }

    inventory.set_immediate_mode();
    inventory.add_metadata("degree", max_degree);
    inventory.add_metadata(
        "visualization_manifest",
        nlohmann::json::object({
            {"flux_moment_dirichlet_field", "scalar_field"},
            {"flux_moment_l2_field", "scalar_field"},
            {"least_squares_field", "scalar_field"},
            {"samples", "point3,density1"},
            {"point_moment_dirichlet_reconstruction", "point3,density1"},
            {"point_moment_l2_reconstruction", "point3,density1"},
            {"flux_moment_dirichlet_reconstruction", "point3,density1"},
            {"flux_moment_l2_reconstruction", "point3,density1"},
            {"least_squares_reconstruction", "point3,density1"},
            {"flux_moment_dirichlet_reconstruction_error", "point3,density1"},
            {"flux_moment_l2_reconstruction_error", "point3,density1"},
            {"least_squares_reconstruction_error", "point3,density1"},
            {"least_squares_reconstruction", "point3,density1"},
            {"least_squares_reconstruction_error", "point3,density1"},
        }));

    std::ofstream(inventory.get_new_asset_path("mesh_info"))
        << creator.serialize_to_json();
    auto m = creator.stored_mesh();
    std::cout << "doing active cell stuff" <<std::endl;
    active_region_index = {};//creator.active_region_index;
    if (active_region_index) {
        active_cells = m->cell_regions().at(*active_region_index);
    } else {
        for (int j = 0; j < m->cell_count(); ++j) {
            active_cells.emplace(j);
        }
    }

    std::cout << "done asdf" << std::endl;
    double val = result["angle"].as<double>() * 180.0 / M_PI;
    double quad_x = result["quad_x"].as<double>();
    double quad_y = result["quad_y"].as<double>();
    Eigen::Vector3d linear_dir(std::cos(val), std::sin(val),0.0);
    Eigen::Vector3d quad_center(quad_x, quad_y,0.0);
    auto linear = [linear_dir](const mtao::Vec3d& p) -> double {
        return linear_dir.dot(p);
    };
    auto quadratic = [linear_dir, quad_center](const mtao::Vec3d& p) -> double {
        return linear_dir.dot(p) + (p - quad_center).array().pow(2).prod();
    };
    auto f = [](const mtao::Vec3d& f) -> double {
        return std::sin(1.0 / ((f.array() - .5).matrix().norm() + 1e-5));
        return f.y();
        return f.x() * f.x();
    };
    std::cout << "Starting to run" <<std::endl;
    for (int j = 0; j < desired_samples_per_cell; ++j) {
        frame_inventory =
            serialization::FrameInventory::for_creation(inventory, j);
        if (use_quad) {
            run(*m, quadratic, max_degree, j);
        } else if (use_linear) {
            run(*m, linear, max_degree, j);
        } else {
            run(*m, f, max_degree, j);
        }
        // auto [A, B] = reconstruct_from_function(*m, f, max_degree, j);
        // std::cout << mtao::eigen::hstack(A, B).transpose() << " " <<
        // std::endl; std::cout << std::endl;
        frame_inventory.reset();
    }
}

