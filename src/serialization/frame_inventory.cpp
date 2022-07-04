#include "vem/serialization/frame_inventory.hpp"
#include <compare>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace vem::serialization {

FrameInventory FrameInventory::for_creation(Inventory& parent, size_t index,
                                            const std::string& format) {
    FrameInventory mine = parent.make_subinventory(fmt::vformat(format, fmt::make_format_args(index)));
    mine.add_metadata("index", index);
    mine.add_metadata("type", "frame");
    return mine;
}
std::unique_ptr<const FrameInventory> FrameInventory::for_ingest(
    const Inventory& parent, const std::string& name) {
    auto ret = std::unique_ptr<const FrameInventory>(
        new FrameInventory(parent.get_subinventory(name)));
    if (ret->metadata("type") != "frame") {
        spdlog::error(
            "Tried to treat {} as a FrameInventory, but it is the wrong type",
            std::string(ret->real_path()));
    }
    return ret;
}

FrameInventory::~FrameInventory() {
    add_metadata("complete", true);
}
FrameInventory::FrameInventory(const Inventory& inventory)
    : Inventory(inventory) {}

FrameInventory::FrameInventory(Inventory&& inventory)
    : Inventory(std::move(inventory)) {}

std::strong_ordering FrameInventory::operator<=>(
    const FrameInventory& o) const {
    return index() <=> o.index();
}
int FrameInventory::index() const { return metadata("index").get<int>(); }
}  // namespace vem::serialization
