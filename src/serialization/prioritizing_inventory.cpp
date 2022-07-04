
#include "vem/serialization/prioritizing_inventory.hpp"

#include <iostream>

namespace vem::serialization {

struct PrioritizingInventoryControlBlock {
    PrioritizingInventory* _current_inventory = nullptr;
};

PrioritizingInventory::PrioritizingInventory(
    Inventory&& inventory,
    std::shared_ptr<PrioritizingInventoryControlBlock> cb)
    : Inventory(std::move(inventory)), _cb(cb) {
    if (bool(_cb)) {
        _previous_inventory = cb->_current_inventory;
    } else {
        _cb = std::make_shared<PrioritizingInventoryControlBlock>();
    }
    _cb->_current_inventory = this;
}
PrioritizingInventory PrioritizingInventory::make_subinventory(
    const std::string& name) {
    return PrioritizingInventory(Inventory::make_subinventory(name), _cb);
}

PrioritizingInventory::~PrioritizingInventory() {
    _cb->_current_inventory = _previous_inventory;
}

PrioritizingInventory& PrioritizingInventory::prioritized_inventory() {
    if (!_cb || _cb->_current_inventory == nullptr) {
        return *this;
    } else {
        return *_cb->_current_inventory;
    }
}

/*
PrioritizingInventoryHandler::PrioritizingInventoryHandler(
    Inventory& inventory, std::shared_ptr<PrioritizingInventoryControlBlock> cb)
    : _inventory(inventory), _cb(cb) {
    if (bool(_cb)) {
        _previous_inventory = cb->_current_inventory;
    } else {
        _cb = std::make_shared<PrioritizingInventoryControlBlock>();
    }
    _cb->_current_inventory = &inventory;
}
PrioritizingInventory PrioritizingInventoryHandler::make_subinventory(
    const std::string& name) {
    return PrioritizingInventory(Inventory::make_subinventory(name), _cb);
}

PrioritizingInventoryHandler::~PrioritizingInventoryHandler() {
    _cb->_current_inventory = _previous_inventory;
}
*/

}  // namespace vem::serialization
