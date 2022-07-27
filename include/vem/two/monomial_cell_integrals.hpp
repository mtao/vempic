#pragma once
#include <map>
#include <mtao/types.hpp>

#include "mesh.hpp"

namespace vem::two {

mtao::VecXd monomial_cell_integrals(const VEMMesh2 &mesh, int index, int max_degree);
mtao::VecXd monomial_cell_integrals(const VEMMesh2 &mesh, int index, int max_degree, const mtao::Vec2d &center);

mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh2 &mesh, int index, double scale, int max_degree);
mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh2 &mesh, int index, double scale, int max_degree, const mtao::Vec2d &center);


}// namespace vem
