#include "vem/visualize/asset_viewer.hpp"

namespace vem::visualize {
bool AssetViewer::load(const Inventory& inv, const std::string& name) {
    const auto& meta = my_inv.asset_metadata(name);
    std::string type = meta["type"];
    std::string storage_type = meta["storage_type"];

    if (!valid_storage_type(storage_type)) {
        spdlog::warn(
            "Viewer[{}] cannot open storage type {}. Valid types are [{}]",
            viewer_type(), storage_type, fmt::join(valid_storage_types(), ";"));
    }
}

bool AssetViewer::valid_type(const std::string& name) const {
    return valid_types().contains(name);
}
bool AssetViewer::valid_storage_type(const std::string& name) const {
    return valid_storage_types().contains(name);
}
}  // namespace vem::visualize

