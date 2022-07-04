#pragma once

#include "vem/serialization/inventory.hpp"
namespace vem::serialization {

void serialize_string(Inventory& inventory, const std::string& name,
        const std::string& file_data);

void serialize_json(Inventory& inventory, const std::string& name,
        const nlohmann::json& js);
}
