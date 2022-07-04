#include "vem/creator2.hpp"

#include <mtao/geometry/mesh/boundary_facets.h>

#include <filesystem>
#include <fstream>
#include <mtao/geometry/bounding_box.hpp>
#include <mtao/geometry/mesh/dual_edges.hpp>
#include <mtao/geometry/mesh/read_obj.hpp>
#include <mtao/json/bounding_box.hpp>
#include <vem/from_grid.hpp>
#include <vem/from_mandoline.hpp>
#include <vem/from_simplicial_matrices.hpp>

namespace {
template <class T>
auto stuff_into_shared_ptr(T &&obj) {
    return std::make_shared<T>(obj);
}
}  // namespace
namespace vem {

std::string VEMMesh2Creator::MeshType2string(MeshType m) {
    switch (m) {
        case MeshType::Grid:
            return "grid";
        case MeshType::TriangleMesh:
            return "triangle_mesh";
        case MeshType::Cutmesh:
            return "cutmesh";
        default:
            return "Unknown";
    }
}
bool VEMMesh2Creator::make_obj_mesh() {
    std::filesystem::path path(mesh_filename);
    spdlog::info("Trying to load mesh [{}] [{}]", mesh_filename,
                 std::string(path));
    if (std::filesystem::exists(path)) {
        auto [V, F] = mtao::geometry::mesh::read_objD(std::string(path));
        _stored_mesh =
            stuff_into_shared_ptr(vem::from_triangle_mesh(V.topRows<2>(), F));
        spdlog::info("Made a mesh with {} {}", V.cols(), F.cols());
    } else {
        spdlog::error("Path not found, cannot create obj mesh");
        return false;
    }

    last_made_mesh = MeshType::TriangleMesh;
    return true;
}

bool VEMMesh2Creator::make_grid_mesh() {
    auto [x, y] = grid_mesh_dimensions;
    if (x < 2 || y < 2) {
        spdlog::info("Grid dimensions too big");
        return false;
    }
    _stored_mesh = stuff_into_shared_ptr(
        vem::from_grid(grid_mesh_bbox.cast<double>(), x, y));
    last_made_mesh = MeshType::Grid;
    return true;
}

bool VEMMesh2Creator::load_boundary_mesh() {
    std::filesystem::path path(mesh_filename);
    spdlog::info("Trying to load boundary mesh [{}] [{}]", mesh_filename,
                 std::string(path));
    if (std::filesystem::exists(path)) {
        auto &[V, E] = _held_boundary_mesh.emplace();
        {
            std::tie(V, E) =
                mtao::geometry::mesh::read_obj2D(std::string(path));
            spdlog::info("V Size {} {}", V.rows(), V.cols());
            std::cout << V << std::endl;
            spdlog::info("E Size {} {}", E.rows(), E.cols());
            std::cout << E << std::endl;
        }
        if (E.size() == 0) {
            spdlog::info(
                "OBJ mesh did not contain edges; falling back to projecting a "
                "triangle mesh and looking for boundaries");
            auto [V_, F] = mtao::geometry::mesh::read_objD(std::string(path));
            V = V_.topRows<2>();

            auto E_ = mtao::geometry::mesh::boundary_facets(F);
            auto DE = mtao::geometry::mesh::dual_edges<3>(F, E_);
            std::set<int> inds;
            for (int j = 0; j < DE.cols(); ++j) {
                if (DE.col(j).minCoeff() < 0) {
                    inds.emplace(j);
                }
            }
            E.resize(2, inds.size());

            for (auto &&[ind, ind2] : mtao::iterator::enumerate(inds)) {
                E.col(ind) = E_.col(ind2);
            }
        }

        spdlog::info("Loaded a boundary mesh with #V={} #E={}", V.cols(),
                     E.cols());
        {
            auto bb = mtao::geometry::bounding_box(V);
            spdlog::info("Bounding box of loaded mesh is {},{} => {},{}",
                         bb.min().x(), bb.min().y(), bb.max().x(),
                         bb.max().y());
        }
        return E.size() > 0 && V.size() > 0;
    } else {
        spdlog::error("Path not found, cannot load boundary mesh");
        return false;
    }
}

bool VEMMesh2Creator::make_mandoline_mesh(bool load_boundary) {
    if (load_boundary) {
        if (!load_boundary_mesh()) {
            spdlog::error(
                "Failed to load the boundary mesh before making a cutmesh");
            return false;
        }
    }
    if (!_held_boundary_mesh) {
        spdlog::error(
            "Need to construct a boundary mesh before trying to cutmesh it");
        return false;
    }
    auto &[V, E] = *_held_boundary_mesh;
    auto [x, y] = grid_mesh_dimensions;
    _stored_mesh = stuff_into_shared_ptr(
        vem::from_mandoline(grid_mesh_bbox.cast<double>(), x, y, V, E, true));
    spdlog::info("Made a cut-cell mesh with #V={} #E={}", V.cols(), E.cols());
    last_made_mesh = MeshType::Cutmesh;
    return true;
}

nlohmann::json VEMMesh2Creator::serialize_to_json() const {
    using json = nlohmann::json;
    json js;

    {
        js["grid"]["bounding_box"] =
            mtao::json::bounding_box2json(grid_mesh_bbox);
        js["grid"]["shape"] = grid_mesh_dimensions;
        auto path = std::filesystem::path(mesh_filename);
        if (std::filesystem::is_regular_file(path)) {
            path = std::filesystem::absolute(path);
            js["mesh"]["filename"] = std::string(path);
            js["mesh"]["input_filename"] = mesh_filename;
        } else {
            js["mesh"]["filename"] = "";
        }
        js["mesh"]["delaminate"] = do_delaminate;
    }
    if (_stored_mesh) {
        js["last_created"] = MeshType2string(last_made_mesh);
    }
    if (active_region_index) {
        js["active_region"] = *active_region_index;
    }
    return js;
}
void VEMMesh2Creator::configure_from_json(const nlohmann::json &js,
                                          bool make_if_available) {
    grid_mesh_bbox =
        js["grid"]["bounding_box"].get<Eigen::AlignedBox<float, 2>>();
    grid_mesh_dimensions = js["grid"]["shape"].get<std::array<int, 2>>();
    mesh_filename = js["mesh"]["filename"];
    spdlog::info("Mesh filename: ", mesh_filename);

    if (js["mesh"].contains("delaminate")) {
        do_delaminate = js["mesh"].at("delaminate");
    } else {
        do_delaminate = false;
    }

    if (make_if_available) {
        if (js.contains("last_created")) {
            const std::string type = js.at("last_created");
            if (type == "grid") {
                make_grid_mesh();
            } else if (type == "triangle_mesh") {
                make_obj_mesh();
            } else if (type == "cutmesh") {
                make_mandoline_mesh();
            }
        }
    }
    if (js.contains("active_region")) {
        active_region_index = js["active_region"];
    } else {
        active_region_index = {};
    }
}
void VEMMesh2Creator::configure_from_json_file(const std::string &filename,
                                               bool make_if_available) {
    std::ifstream fs(filename);
    nlohmann::json j;
    fs >> j;
    configure_from_json(j, make_if_available);
}
}  // namespace vem

