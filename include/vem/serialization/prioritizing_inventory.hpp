#pragma once

#include "vem/serialization/inventory.hpp"

namespace vem::serialization {

// An inventory object that's designed aroudn building a tree of inventories
// using RAII that way when we declare a sub-inventory

class PrioritizingInventoryControlBlock;

class PrioritizingInventory : public Inventory {
   public:
    PrioritizingInventory(
        Inventory&& inventory,
        std::shared_ptr<PrioritizingInventoryControlBlock> cb = nullptr);
    ~PrioritizingInventory();

    PrioritizingInventory make_subinventory(const std::string& name);
    PrioritizingInventory& prioritized_inventory();

   private:
    std::shared_ptr<PrioritizingInventoryControlBlock> _cb = nullptr;
    PrioritizingInventory* _previous_inventory;
};

/*
class PrioritizingInventoryHandler {
    PrioritizingInventoryHandler(
        Inventory& inventory,
        std::shared_ptr<PrioritizingInventoryControlBlock> cb = nullptr);

    // should I really allow for this?
    // PrioritizingInventoryHandler(PrioritizingInventory& inventory);
    ~PrioritizingInventoryHandler();

    PrioritizingInventory make_subinventory(const std::string& name);

    std::shared_ptr<PrioritizingInventoryControlBlock> _cb = nullptr;
    Inventory& _inventory;
    Inventory* _previous_inventory = nullptr;
};
*/
}  // namespace vem::serialization
