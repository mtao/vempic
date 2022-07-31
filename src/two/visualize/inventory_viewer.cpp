#include "vem/two/visualize/inventory_viewer.hpp"

#include <imgui.h>

#include "vem/two/creator.hpp"
#include "vem/serialization//serialize_eigen.hpp"
namespace {

std::vector<std::string> get_csvalues(const std::string& str) {
    std::stringstream ss(str);
    std::vector<std::string> ret;
    std::string line;
    while (!ss.eof()) {
        getline(ss, line, ',');
        ret.emplace_back(std::move(line));
    }
    return ret;
}
}  // namespace

// borrowed from
// https://eliasdaler.github.io/using-imgui-with-sfml-pt2/#combobox-listbox
namespace ImGui {
auto vector_getter = [](void* vec, int idx, const char** out_text) {
    auto& vector = *static_cast<std::vector<std::string>*>(vec);
    if (idx < 0 || idx >= static_cast<int>(vector.size())) {
        return false;
    }
    *out_text = vector.at(idx).c_str();
    return true;
};

bool Combo(const char* label, int* currIndex,
           std::vector<std::string>& values) {
    if (values.empty()) {
        return false;
    }
    return Combo(label, currIndex, vector_getter, static_cast<void*>(&values),
                 values.size());
}

bool ListBox(const char* label, int* currIndex,
             std::vector<std::string>& values) {
    if (values.empty()) {
        return false;
    }
    return ListBox(label, currIndex, vector_getter, static_cast<void*>(&values),
                   values.size());
}

}  // namespace ImGui

namespace vem::two::visualize {

template <typename DataMap>
auto InventoryViewer::update_data(const std::string& current,
                                  const DataMap& data) const
    -> std::optional<typename DataMap::mapped_type> {
    // spdlog::info("Searching for {}", current);
    if (auto it = data.find(current); it != data.end()) {
        return it->second;
    } else {
        spdlog::info("Data for {} was not found!", current);
        return {};
    }
}
InventoryViewer::InventoryViewer(
    const std::shared_ptr<serialization::Inventory>& inventory,
    mtao::opengl::Object2D* parent, Magnum::SceneGraph::DrawableGroup2D* group)
    : mtao::opengl::Object2D{parent},
      Magnum::SceneGraph::Drawable2D{*this, group},
      _inventory(inventory),
      scalar_field_viewer(&background_drawables)

{
    boundary_mesh.setParent(this);
    point_sample_mesh.setParent(this);
    point_density_mesh.setParent(this);
    point_vector_mesh.setParent(this);
    scalar_field_viewer.setParent(this);

    if (boundary_drawable == nullptr) {
        boundary_drawable = new mtao::opengl::MeshDrawable{
            boundary_mesh, flat_shader, foreground_drawables};
        boundary_drawable->data().color = Magnum::Color4(0, 0, 0, 1);
        boundary_drawable->deactivate();
        boundary_drawable->line_width = 5;
    }

    if (point_vector_drawable == nullptr) {
        point_vector_drawable =
            new mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>{
                point_vector_mesh, vf_shader, foreground_drawables};
        point_vector_drawable->deactivate();
    }

    if (point_sample_node_drawable == nullptr) {
        point_sample_node_drawable =
            new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>{
                point_sample_mesh, flat_shader, foreground_drawables};
        point_sample_node_drawable->deactivate();
        point_sample_node_drawable->point_size = 5;
    }

    if (point_density_node_drawable == nullptr) {
        point_density_node_drawable =
            new mtao::opengl::MeshDrawable<Magnum::Shaders::VertexColor2D>{
                point_density_mesh, vertex_color_shader, foreground_drawables};
        point_density_node_drawable->deactivate();
        point_density_node_drawable->point_size = 5;
    }

    reload();
    load_current_frame();
}
InventoryViewer::~InventoryViewer() {}

void InventoryViewer::gui() {
    ImGui::Begin("Inventory Vis");
    if (ImGui::Button("Reload")) {
        reload();
        load_current_frame();
    }
    ImGui::Checkbox("Draw background(scalar field)", &draw_background);
    {
        auto& vec = assets_by_type["scalar_field"];
        if (vec.empty()) {
            ImGui::Text("No Scalar Fields");
        } else {
            bool a = bool(current_scalar_field_index);
            if (ImGui::Checkbox("Show scalar fields", &a)) {
                if (a) {
                    current_scalar_field_index = 0;
                } else {
                    current_scalar_field_index = {};
                }
            }
            if (bool(current_scalar_field_index) &&
                ImGui::ListBox("Scalar Field", &*current_scalar_field_index,
                               vec)) {
                update_scalar_field();
            }
        }
    }
    {
        auto& vec = assets_by_type["velocity2"];
        if (vec.empty()) {
            ImGui::Text("No Point Vectors");
        } else {
            bool a = bool(current_point_vector_index);
            if (ImGui::Checkbox("Show point vectors", &a)) {
                if (a) {
                    current_point_vector_index = 0;
                } else {
                    point_vector_drawable->deactivate();
                    current_point_vector_index = {};
                }
                update_point_vector();
            }
            if (bool(current_point_vector_index) &&
                ImGui::ListBox("Point Vectors", &*current_point_vector_index,
                               vec)) {
                update_point_vector();
            }
        }
    }
    {
        auto& vec = assets_by_type["density1"];
        if (vec.empty()) {
            ImGui::Text("No Density Points");
        } else {
            bool a = bool(current_point_density_index);
            if (ImGui::Checkbox("Show point densities", &a)) {
                if (a) {
                    current_point_density_index = 0;
                } else {
                    point_density_node_drawable->deactivate();
                    current_point_density_index = {};
                }
                update_point_density();
            }
            if (bool(current_point_density_index) &&
                ImGui::ListBox("Point Densities", &*current_point_density_index,
                               vec)) {
                update_point_density();
            }
        }
    }
    {
        auto& vec = assets_by_type["point2"];
        if (vec.empty()) {
            ImGui::Text("No Particles");
        } else {
            bool a = bool(current_point_index);
            if (ImGui::Checkbox("Show particles", &a)) {
                if (a) {
                    current_point_index = 0;
                } else {
                    current_point_index = {};

                    point_sample_node_drawable->deactivate();
                }
                update_point();
            }
            if (bool(current_point_index) &&
                ImGui::ListBox("Point samples", &*current_point_index, vec)) {
                update_point();
            }
        }
    }

    int new_frame = frame_index;
    if (ImGui::InputInt("Frame index", &new_frame)) {
        set_current_frame(new_frame);
    }

    ImGui::End();
    ImGui::Begin("Scalar Field Settings");
    if (scalar_field_viewer.gui()) {
        update_point_density();
    }
    ImGui::End();
}

bool InventoryViewer::reload() {
    if (!_inventory) {
        spdlog::warn(
            "Cant reload an inventory if InventoryViewer does not have "
            "one");
        return false;
    }
    _inventory->reload();
    size_t old_size = _frame_inventories.size();

    assets_by_type.clear();
    for (auto&& [name, expected_type] : visualization_manifest().items()) {
        std::string str = expected_type.get<std::string>();

        for (auto&& t : get_csvalues(str)) {
            if (t == "scalar_field1") {
                assets_by_type["scalar_field"].emplace_back(name);
                increased_degree_fields.emplace(name);
            } else {
                assets_by_type[t].emplace_back(name);
            }
        }
    }
    for (auto&& [asset, names] : assets_by_type) {
        spdlog::info("Asset type [{}] has {} objects", asset, names.size());
    }

    for (auto&& [_, ass] : assets_by_type) {
        std::sort(ass.begin(), ass.end());
        ass.erase(std::unique(ass.begin(), ass.end()), ass.end());
    }
    _frame_inventories.clear();
    for (auto&& subinventory_name : _inventory->get_subinventory_names()) {
        spdlog::info("Checkout out subinventory {}",
                     std::string(subinventory_name));
        try {
            const serialization::Inventory sub_inv =
                _inventory->get_subinventory(subinventory_name);
            if (sub_inv.metadata()["type"] == "frame" &&
                sub_inv.metadata()["complete"].get<bool>()) {
                // i'll just recreate the subinventory and toss the old
                // one, it should be cheap anyway, whatever
                _frame_inventories.emplace_back(
                    serialization::FrameInventory::for_ingest(
                        *_inventory, subinventory_name));
            }
        } catch (const std::exception& e) {
            spdlog::info("Stopped checking subinventory at {} due to: {}",
                         subinventory_name, e.what());
        }
    }
    std::sort(_frame_inventories.begin(), _frame_inventories.end(),
              [](auto&& a, auto&& b) -> bool { return *a < *b; });
    for (auto&& [idx, inv] : mtao::iterator::enumerate(_frame_inventories)) {
        // if we fail to get the index or we fail to get the right index
        // we fall through to an error about missing a frame
        try {
            if (idx == inv->index()) {
                continue;
            }
        } catch (const std::exception& e) {
            spdlog::warn(
                "Exception caught getting index for frame stored at {}",
                std::string(inv->real_path()));
        }

        spdlog::error("Missing frame {}, giving up on loading (got {} insetad)",
                      idx, inv->index());
        _frame_inventories.resize(idx);
        break;
    }

    spdlog::info("Collected {} frames", _frame_inventories.size());
    create_mesh();
    frame_index =
        std::clamp<int>(frame_index, 0, _frame_inventories.size() - 1);

    return old_size == _frame_inventories.size();
}
bool InventoryViewer::set_current_frame(int index) {
    int old_frame = frame_index;
    frame_index = std::clamp<int>(index, 0, _frame_inventories.size() - 1);
    if (old_frame != frame_index) {
        load_current_frame();
    }
    return frame_index == index;
}

bool InventoryViewer::load_current_frame() {
    if (frame_index < 0) {
        spdlog::debug("Frame index {} was too low. setting it to 0 ",
                      frame_index);
        frame_index = 0;
        return false;
    }
    if (frame_index >= _frame_inventories.size()) {
        spdlog::debug(
            "Frame index {} was too high. setting it to teh max valid "
            "frame "
            "({})",
            frame_index, _frame_inventories.size() - 1);
        frame_index = _frame_inventories.size() - 1;
        return false;
    }
    return load_frame(frame_index);
}

void InventoryViewer::draw(const Magnum::Matrix3& transformationMatrix,
                           Magnum::SceneGraph::Camera2D& camera) {
    if (draw_background) {
        camera.draw(background_drawables);
    }
    camera.draw(drawables);
    camera.draw(foreground_drawables);
}

void InventoryViewer::update_vis() {
    update_scalar_field();
    update_point_vector();
    update_point_density();
    update_point();
}

void InventoryViewer::clear_vis() {
    point_sample_node_drawable->deactivate();
    point_vector_drawable->deactivate();
}
bool InventoryViewer::load_frame(int index) {
    if (index < 0) {
        spdlog::debug("Frame index {} was too low. setting it to 0 ", index);
        clear_vis();
        return false;
    }
    if (index >= _frame_inventories.size()) {
        spdlog::debug(
            "Frame index {} was too high. setting it to teh max valid "
            "frame "
            "({})",
            index, _frame_inventories.size() - 1);
        clear_vis();
        return false;
    }
    spdlog::info("Loading frame {}", index);

    const serialization::FrameInventory& my_inv =
        *_frame_inventories[frame_index];

    point_vector_data.clear();
    scalar_field_data.clear();
    point_density_data.clear();
    point_data.clear();
    for (auto&& [name, expected_type] : visualization_manifest().items()) {
        try {
            std::cout << "Trying" << std::endl;
            const auto& meta = my_inv.asset_metadata(name);
            std::cout << "getting path" << std::endl;
            const auto& path = my_inv.get_asset_path(name);
            std::cout << "Checking storage type" << std::endl;
            std::string storage_type = meta["storage_type"];
            std::cout << "Checking type" << std::endl;
            std::string type = meta["type"];
            if (type != expected_type) {
                spdlog::warn("Frame {} has a {} of type {} expected {}",
                             frame_index, std::string(name), std::string(type),
                             std::string(expected_type));
            }
            // spdlog::info("Loading frame {} has a {} of type {} expected {}",
            //             frame_index, std::string(name), std::string(type),
            //             std::string(expected_type));
            if (type == "point2") {
                // particles

                auto V = serialization::deserialize_points2(my_inv, name)
                             .cast<float>()
                             .eval();
                point_data[name] = V;
            } else if (type == "point2,velocity2") {
                // particles with velocities, gotta split it up

                point_vector_data[name] =
                    serialization::deserialize_points4(my_inv, name)
                        .cast<float>();
            } else if (type == "point2,velocity2,density1") {
                // particles with velocities, gotta split it up

                mtao::ColVecs5d dat =
                    serialization::deserialize_points5(my_inv, name);

                point_vector_data[name] = dat.topRows<4>().cast<float>();
                auto& ddat = point_density_data[name];
                ddat.resize(3, dat.cols());
                ddat.topRows<2>() = dat.topRows<2>().cast<float>();
                ddat.row(2) = dat.row(4).cast<float>();
                // spdlog::info("point vector data has {} with data size {} {}
                // from {} {}",

                //        name,
                //        point_vector_data[name].rows(),
                //        point_vector_data[name].cols(),
                //        dat.rows(), dat.cols()
                //        );

            } else if (type == "point2,density1") {
                // particles with velocities, gotta split it up

                std::cout << name << std::endl;
                mtao::ColVecs3d dat =
                    serialization::deserialize_points3(my_inv, name);

                point_density_data[name] = dat.cast<float>();
            } else if (type == "scalar_field") {
                scalar_field_data[name] =
                    serialization::deserialize_VecXd(my_inv, name)
                        .cast<float>();
            } else if (type == "scalar_field1") {
                scalar_field_data[name] =
                    serialization::deserialize_VecXd(my_inv, name)
                        .cast<float>();
            } else {
                spdlog::warn(
                    "Unhandled type presented when loading frame: {} of type "
                    "{}",
                    name, type);
            }
        } catch (const std::exception& e) {
            spdlog::info("Encountered an error loading {} on frame {}: {}",
                         name, frame_index, e.what());
        }
    }
    for (auto&& [name, dat] : point_vector_data) {
        if (!point_data.contains(name)) {
            point_data[name] = dat.topRows<2>();
        }
    }

    update_vis();
    return true;
}

void InventoryViewer::create_mesh() {
    if (!_inventory) {
        spdlog::warn(
            "Cant make mesh without an inventory"
            "one");
        return;
    }
    // create the mesh object
    std::filesystem::path config_path = _inventory->get_asset_path("mesh_info");
    VEMMesh2Creator creator;
    creator.configure_from_json_file(config_path, true);

    _mesh = creator.stored_mesh();

    scalar_field_viewer.set_mesh(*_mesh);

    const auto& meta = _inventory->metadata();
    if (meta.contains("per_degrees")) {
        const auto& degs = meta.at("per_degrees");
        monomial_indexer = std::make_shared<MonomialBasisIndexer>(
            *_mesh, degs.get<std::vector<size_t>>());
        monomial_indexer_up = std::make_shared<MonomialBasisIndexer>(
            monomial_indexer->antiderivative_indexer());

    } else if (meta.contains("degree")) {
        int degree = meta.at("degree");
        monomial_indexer =
            std::make_shared<MonomialBasisIndexer>(*_mesh, degree);

        monomial_indexer_up = std::make_shared<MonomialBasisIndexer>(
            monomial_indexer->antiderivative_indexer());
    } else {
        spdlog::error("Metadta does not include a degree");
    }
    // spdlog::info("{}", fmt::join(monomial_indexer_up->degrees(), ","));
    spdlog::info("Scalar field viewer should have a monomial indexer now {}",
                 bool(monomial_indexer));
    scalar_field_viewer.set_monomial_indexer(monomial_indexer);

    try {
        if (_inventory->has_subinventory("mesh")) {
            spdlog::info("Loading mesh boundary");
            const auto subinv = _inventory->get_subinventory("mesh");
            auto edges = subinv.metadata().at("boundary_edges");
            if (edges.size() > 0) {
                mtao::ColVecs2i E(2, edges.size());
                for (auto&& [idx, edge_index] :
                     mtao::iterator::enumerate(edges)) {
                    E.col(idx) = _mesh->E.col(int(edge_index));
                }
                spdlog::info("Got {} boundary edges for the active region",
                             E.cols());
                boundary_mesh.setVertexBuffer(_mesh->V.cast<float>().eval());
                boundary_mesh.setEdgeBuffer(E.cast<unsigned int>().eval());
                if (boundary_drawable) {
                    boundary_drawable->activate_edges();
                }
            }
        }

    } catch (const std::exception& e) {
        spdlog::warn("Failed to obtain boundary edges from mesh, got error: ",
                     e.what());
        boundary_drawable->deactivate();
    }
}

const nlohmann::json& InventoryViewer::visualization_manifest() const {
    const static nlohmann::json empty;
    if (!_inventory) {
        spdlog::warn(
            "Cant read manifest of inventory if InventoryViewer does "
            "not "
            "have "
            "one");
        return empty;
    }
    return _inventory->metadata("visualization_manifest");
}

void InventoryViewer::update_point() {
    auto cur = current_point();
    if (!cur.empty()) {
        auto dat = update_data(cur, point_data);

        if (dat) {
            auto P = dat->topRows<2>().cast<float>().eval();
            point_sample_mesh.setVertexBuffer(P);
            point_sample_node_drawable->activate_points();
            return;
        }
    }

    point_sample_node_drawable->deactivate();
}
void InventoryViewer::update_point_density() {
    auto cur = current_point_density();
    if (!cur.empty()) {
        auto dat = update_data(cur, point_density_data);

        if (dat) {
            auto P = dat->topRows<2>().cast<float>().eval();
            point_density_mesh.setVertexBuffer(P);
            auto D = dat->row(2);
            mtao::ColVecs4f C(4, D.size());
            for (auto [idx, d] : mtao::iterator::enumerate(D)) {
                auto c = C.col(idx);
                c = scalar_field_viewer.get_color(d);
            }
            point_density_mesh.setColorBuffer(C);

            point_density_node_drawable->activate_points();
            return;
        }
    }

    point_density_node_drawable->deactivate();
}

void InventoryViewer::update_point_vector() {
    auto cur = current_point_vector();
    if (!cur.empty()) {
        auto dat = update_data(cur, point_vector_data);

        if (dat) {
            auto P = dat->topRows<2>().cast<float>().eval();
            auto V = dat->bottomRows<2>().cast<float>().eval();
            point_vector_mesh.setVertexBuffer(P);
            point_vector_mesh.setVFieldBuffer(V);
            point_vector_drawable->activate_points();
            return;
        }
    }

    point_vector_drawable->deactivate();
}

void InventoryViewer::update_scalar_field() {
    auto cur = current_scalar_field();
    if (!cur.empty()) {
        auto dat = update_data(cur, scalar_field_data);
        if (dat) {
            if (increased_degree_fields.contains(cur)) {
                spdlog::info("using an increased guy");
                scalar_field_viewer.set_monomial_indexer(monomial_indexer_up);
            } else {
                scalar_field_viewer.set_monomial_indexer(monomial_indexer);
            }
            scalar_field_viewer.set_coefficients(dat->cast<double>());
        }
    }
}

const std::string& InventoryViewer::get_asset_name_from_index(
    const std::string& name, const std::optional<size_t>& index) const {
    const static std::string empty = "";
    if (index) {
        if (auto it = assets_by_type.find(name); it != assets_by_type.end()) {
            const auto& vec = it->second;
            if (*index < vec.size()) {
                return vec.at(*index);
            }
        }
    }
    return empty;
}

const std::string& InventoryViewer::current_point() const {
    const static std::string s = "point2";
    return get_asset_name_from_index(s, current_point_index);
}
const std::string& InventoryViewer::current_point_vector() const {
    const static std::string s = "velocity2";
    return get_asset_name_from_index(s, current_point_vector_index);
}
const std::string& InventoryViewer::current_point_density() const {
    const static std::string s = "density1";
    return get_asset_name_from_index(s, current_point_density_index);
}
const std::string& InventoryViewer::current_scalar_field() const {
    const static std::string s = "scalar_field";
    return get_asset_name_from_index(s, current_scalar_field_index);
}
}  // namespace vem::visualize
