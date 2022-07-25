#include "vem/creator3.hpp"

#include <mtao/logging/stopwatch.hpp>
#include <igl/read_triangle_mesh.h>
#include <mtao/geometry/mesh/boundary_facets.h>

#include <filesystem>
#include <fstream>
#include <mtao/geometry/bounding_box.hpp>
#include <mtao/geometry/mesh/dual_edges.hpp>
#include <mtao/geometry/mesh/read_obj.hpp>
#include <mtao/json/bounding_box.hpp>
#include <tuple>
#include <vem/from_grid3.hpp>
#include <vem/from_mandoline3.hpp>
#include <vem/from_simplicial_matrices.hpp>

namespace {
template <class T>
auto stuff_into_shared_ptr(T &&obj) {
    return std::make_shared<T>(obj);
}
}  // namespace
namespace vem {

std::string VEMMesh3Creator::MeshType2string(MeshType m) {
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

bool VEMMesh3Creator::make_grid_mesh() {
    auto [x, y, z] = grid_mesh_dimensions;
    if (x < 2 || y < 2 || z < 2) {
        spdlog::info("Grid dimensions too big");
        return false;
    }
    _stored_mesh = stuff_into_shared_ptr(
        vem::from_grid(grid_mesh_bbox.cast<double>(), x, y, z));
    last_made_mesh = MeshType::Grid;
    return true;
}

bool VEMMesh3Creator::load_boundary_mesh() {
    std::filesystem::path path(mesh_filename);
    spdlog::info("Trying to load boundary mesh [{}] [{}]", mesh_filename,
                 std::string(path));
    if (std::filesystem::exists(path)) {
        Eigen::MatrixXd VV;
        Eigen::MatrixXi FF;
        igl::read_triangle_mesh(mesh_filename, VV, FF);
        auto &[V, F] = _held_boundary_mesh.emplace();
        V = VV.transpose();
        F = FF.transpose();

        spdlog::info("V Size {} {}", V.rows(), V.cols());
        spdlog::info("F Size {} {}", F.rows(), F.cols());

        spdlog::info("Loaded a boundary mesh with #V={} #F={}", V.cols(),
                     F.cols());
        {
            auto bb = mtao::geometry::bounding_box(V);
            spdlog::info("Bounding box of loaded mesh is {},{},{} => {},{},{}",
                         bb.min().x(), bb.min().y(), bb.min().z(), bb.max().x(),
                         bb.max().y(), bb.max().z());
        }
        return F.size() > 0 && V.size() > 0;
    } else {
        spdlog::error("Path not found, cannot load boundary mesh");
        return false;
    }
}

bool VEMMesh3Creator::make_mandoline_mesh(bool load_boundary) {
    std::filesystem::path path(mesh_filename);
    if (mesh_filename.ends_with(".cutmesh")) {
        spdlog::info("Loading cutmesh {}", mesh_filename);
        auto mptr = stuff_into_shared_ptr(vem::from_mandoline(
            mandoline::CutCellMesh<3>::from_file(mesh_filename)));
        const auto &ccm = mptr->_ccm;
        mptr->_ccm.write("/tmp/vemsim.cutmesh");
        _stored_mesh = mptr;

        spdlog::info("Loaded a cutmesh from a cutmesh file");
    } else {
    auto sw = mtao::logging::hierarchical_stopwatch("MandolineGeometry");
        spdlog::info("Constructing a cutmesh");
        if (load_boundary) {
            if (!load_boundary_mesh()) {
                spdlog::error(
                    "Failed to load the boundary mesh before making a cutmesh");
                return false;
            }
        }
        if (!_held_boundary_mesh) {
            spdlog::error(
                "Need to construct a boundary mesh before trying to cutmesh "
                "it");
            return false;
        }
        spdlog::info("Fetching boundary mesh info");
        auto &[V, F] = *_held_boundary_mesh;
        auto [x, y, z] = grid_mesh_dimensions;
        spdlog::info("grid dimensions {} {} {}", x, y, z);
        for (int j = 0; j < V.cols(); ++j) {
            if (!grid_mesh_bbox.contains(V.col(j).cast<float>())) {
                spdlog::error(
                    "The effectors lie outside of the domain bounding box, "
                    "exiting");
                return false;
            }
        }
        spdlog::info("Running mandoline");
        std::cout << grid_mesh_bbox.min().transpose() << " => "
                  << grid_mesh_bbox.max().transpose() << std::endl;
        std::cout << F.cols() << " " << V.cols() << " " << adaptive_grid_level
                  << std::endl;
        auto mptr = stuff_into_shared_ptr(vem::from_mandoline(
            grid_mesh_bbox.cast<double>(), x, y, z, V, F, adaptive_grid_level));
        spdlog::info("Made a cut-cell mesh with #V={} #F={}", V.cols(),
                     F.cols());
        spdlog::info("Writing time");
        mptr->_ccm.write("/tmp/vemsim.cutmesh");
        _stored_mesh = mptr;
    }

    last_made_mesh = MeshType::Cutmesh;
    return true;
}

nlohmann::json VEMMesh3Creator::serialize_to_json() const {
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
    }
    js["adaptive_grid_level"] = adaptive_grid_level;
    if (_stored_mesh) {
        js["last_created"] = MeshType2string(last_made_mesh);
    }
    return js;
}
void VEMMesh3Creator::configure_from_json(const nlohmann::json &js,
                                          bool make_if_available) {
    std::cout << js << std::endl;
    grid_mesh_bbox =
        js["grid"]["bounding_box"].get<Eigen::AlignedBox<float, 3>>();
    grid_mesh_dimensions = js["grid"]["shape"].get<std::array<int, 3>>();
    bool force_cutmesh = js.contains("cutmesh");
    if (force_cutmesh) {
        spdlog::error("Forcing cutmesh");
        mesh_filename = js["cutmesh"];
    } else {
        mesh_filename = js["mesh"]["filename"];
    }
    spdlog::info("Mesh filename: ", mesh_filename);

    if (js.contains("adaptive_grid_level")) {
        adaptive_grid_level = js["adaptive_grid_level"];
    } else {
        adaptive_grid_level = 0;
    }
    if (make_if_available) {
        if (force_cutmesh) {
            make_mandoline_mesh();
        } else if (js.contains("last_created")) {
            const std::string type = js.at("last_created");
            if (type == "grid") {
                make_grid_mesh();
            } else if (type == "triangle_mesh") {
                // make_obj_mesh();
            } else if (type == "cutmesh") {
                make_mandoline_mesh();
            }
        }
    }
}
void VEMMesh3Creator::configure_from_json_file(const std::string &filename,
                                               bool make_if_available) {
    std::ifstream fs(filename);
    nlohmann::json j;
    fs >> j;
    configure_from_json(j, make_if_available);
}
}  // namespace vem

