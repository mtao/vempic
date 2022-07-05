#pragma once
#include <map>
#include <mtao/types.hpp>

#include "vem/mesh.hpp"

namespace vem {


mtao::VecXd monomial_cell_integrals(const VEMMesh3 &mesh, int index, int max_degree);
mtao::VecXd monomial_cell_integrals(const VEMMesh3 &mesh, int index, int max_degree, const mtao::Vec3d &center);

mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh3 &mesh, int index, double scale, int max_degree);
mtao::VecXd scaled_monomial_cell_integrals(const VEMMesh3 &mesh, int index, double scale, int max_degree, const mtao::Vec3d &center);
}// namespace vem
