#pragma once
#include <mtao/types.hpp>

namespace vem::utils {
class InsideGeometryPredicate {
   public:
    virtual bool operator()(const mtao::Vec3d& p) const = 0;
};
}  // namespace vem::utils
