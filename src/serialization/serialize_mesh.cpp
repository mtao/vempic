#include "vem/serialization/serialize_mesh.hpp"
#include "vem/two/mesh.hpp"
#include "vem/three/mesh.hpp"

#include "vem/serialization/serialize_eigen.hpp"
namespace vem::serialization {

void serialize_mesh(Inventory& inventory, const std::string& name,
                    const two::VEMMesh2& mesh) {
    Inventory subinv = inventory.make_subinventory(name);
    subinv.add_metadata("storage_type", "vem_mesh");
    subinv.add_metadata("dimension", 2);

    Inventory vertex_subinv = subinv.make_subinventory("vertices");
    serialize_points2(subinv, "vertices", mesh.V);
    serialize_points2(subinv, "cell_centers", mesh.C);
    serialize_points2(subinv, "edges", mesh.E.cast<double>());

    subinv.add_metadata("regions", mesh.cell_regions());
    subinv.add_metadata("boundary_edges", mesh.boundary_edge_indices());
    auto& cells = subinv.metadata("cells") = nlohmann::json::array();
    spdlog::warn("Writing cell data");
    for (auto&& c : mesh.face_loops()) {
        cells.push_back(static_cast<std::vector<int>&>(c));
    }
}

void serialize_mesh(Inventory& inventory, const std::string& name,
                    const three::VEMMesh3& mesh) {
    Inventory subinv = inventory.make_subinventory(name);
    subinv.add_metadata("storage_type", "vem_mesh");
    subinv.add_metadata("dimension", 3);

    Inventory vertex_subinv = subinv.make_subinventory("vertices");
    serialize_points3(subinv, "vertices", mesh.V);
    serialize_points3(subinv, "cell_centers", mesh.C);
    // serialize_points3(subinv, "edges", mesh.E.cast<double>());

    subinv.add_metadata("regions", mesh.cell_regions());
    // subinv.add_metadata("boundary_edges", mesh.boundary_edge_indices());
    spdlog::warn("MTAO needs to add boundary data to the mesh");
    auto& cells = subinv.metadata("cells") = nlohmann::json::array();
    // for (auto&& c : mesh.face_loops()) {
    //    cells.push_back(static_cast<std::vector<int>&>(c));
    //}
}
}  // namespace vem::serialization
