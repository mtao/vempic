#pragma once
#include "vem/cell.hpp"

namespace vem {
class BoundaryMomentVEM2Cell : public VEM2Cell {
   public:
    mtao::MatXd projection_l2() const;
    mtao::MatXd projection_dirichlet() const;

   private:
};

}  // namespace vem
