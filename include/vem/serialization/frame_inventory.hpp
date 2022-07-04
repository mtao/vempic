#pragma once
#include <compare>

#include "vem/serialization/inventory.hpp"

namespace vem::serialization {
class FrameInventory : public Inventory {
   public:
    static FrameInventory for_creation(Inventory& parent, size_t index,
                                       const std::string& format = "frame_{}");
    static std::unique_ptr<const FrameInventory> for_ingest(
        const Inventory& parent, const std::string& name);

    std::strong_ordering operator<=>(const FrameInventory& o) const;

    int index() const;
    FrameInventory(FrameInventory&&) = default;
    FrameInventory& operator=(FrameInventory&&) = default;
    ~FrameInventory();

   private:
    using Inventory::Inventory;
    FrameInventory(const Inventory& inventory);
    FrameInventory(Inventory&& inventory);
};
}  // namespace vem::serialization
