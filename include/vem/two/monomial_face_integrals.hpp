#pragma one
#include <map>
#include <mtao/types.hpp>

#include "mesh.hpp"

namespace vem::two {

mtao::VecXd monomial_face_integrals(const VEMMesh2 &mesh, int cell_index,
                                    int edge_index, int max_degree);

mtao::VecXd monomial_face_integrals(const VEMMesh2 &mesh, int index,
                                    int max_degree, const mtao::Vec2d &center);

mtao::VecXd scaled_monomial_face_integrals(const VEMMesh2 &mesh, int cell_index,
                                           int edge_index, double scale,
                                           int max_degree);
mtao::VecXd scaled_monomial_face_integrals(const VEMMesh2 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec2d &center);

}  // namespace vem
