#include <tbb/parallel_for.h>

#include <mtao/geometry/bounding_box.hpp>
#include <mtao/geometry/mesh/write_obj.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <mtao/json/bounding_box.hpp>
#include <vem/cell3.hpp>
#include <vem/creator3.hpp>
#include <vem/flux_moment_cell3.hpp>
#include <vem/flux_moment_indexer3.hpp>
#include <vem/from_mandoline3.hpp>
#include <vem/monomial_basis_indexer_new.hpp>
#include <vem/monomial_cell_integrals.hpp>
#include <vem/utils/face_boundary_facets.hpp>

#include "mtao/eigen/sparse_block_diagonal_repmats.hpp"
#include "vem/mesh.hpp"

int main(int argc, char* argv[]) {
    std::string js_path = argv[1];
    nlohmann::json js;
    std::ifstream(argv[1]) >> js;

    int resolution = js["domain_resolution"];
    auto bbox = js["bounding_box"].get<Eigen::AlignedBox<double, 3>>();
    vem::VEMMesh3Creator creator;

    creator.load_boundary_mesh();

    creator.grid_mesh_bbox = bbox.cast<float>();
    creator.grid_mesh_dimensions[0] = resolution;
    creator.grid_mesh_dimensions[1] = resolution;
    creator.grid_mesh_dimensions[2] = resolution;

    if (js.contains("collision_mesh")) {
        creator.mesh_filename = js["collision_mesh"];
        creator.make_mandoline_mesh();

    } else {
        creator.make_grid_mesh();
    }

    {
        const auto& mesh = *creator.stored_mesh();

        auto regions = mesh.cell_regions();
        spdlog::info("{} cells with {} regions", mesh.cell_count(),
                     regions.size());
        spdlog::info("Cells in region: {}", regions[0]);
        auto [V, F, cbm] = mesh.collision_mesh();
        // auto [V, F, cbm] = mesh.collision_mesh(regions[0]);
        mtao::geometry::mesh::write_objD(V, F, argv[2]);
    }
    return 0;
}
