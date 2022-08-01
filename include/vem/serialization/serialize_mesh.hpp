#pragma once

#include "vem/serialization/inventory.hpp"
namespace vem {
    namespace two {
    class VEMMesh2;
    }
    namespace three {
    class VEMMesh3;
    }
}
namespace vem::serialization {

void serialize_mesh(Inventory& inventory, const std::string& name,
                    const two::VEMMesh2& mesh);
void serialize_mesh(Inventory& inventory, const std::string& name,
                    const three::VEMMesh3& mesh);
}  // namespace vem::serialization
