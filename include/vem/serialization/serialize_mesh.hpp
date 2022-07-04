#pragma once

#include "vem/mesh.hpp"
#include "vem/serialization/inventory.hpp"
namespace vem::serialization {

void serialize_mesh(Inventory& inventory, const std::string& name,
                    const VEMMesh2& mesh);
void serialize_mesh(Inventory& inventory, const std::string& name,
                    const VEMMesh3& mesh);
}  // namespace vem::serialization
