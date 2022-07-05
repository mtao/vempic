#pragma once
#include <map>
#include <mtao/types.hpp>

#include "mesh.hpp"

namespace vem {


mtao::VecXd monomial_face_integrals(const VEMMesh3 &mesh, int cell_index,
                                    int index, int max_degree);
mtao::VecXd monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                    int max_degree, const mtao::Vec3d &center);

mtao::VecXd scaled_monomial_face_integrals(const VEMMesh3 &mesh, int cell_index,
                                           int edge_index, double scale,
                                           int max_degree);
mtao::VecXd scaled_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                           double scale, int max_degree,
                                           const mtao::Vec3d &center);

mtao::VecXd face_monomial_face_integrals(const VEMMesh3 &mesh, int face_index,
                                         int max_degree);
mtao::VecXd face_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                         int max_degree,
                                         const mtao::Vec3d &center);

mtao::VecXd scaled_face_monomial_face_integrals(const VEMMesh3 &mesh,
                                                int face_index, double scale,
                                                int max_degree);
mtao::VecXd scaled_face_monomial_face_integrals(const VEMMesh3 &mesh, int index,
                                                double scale, int max_degree,
                                                const mtao::Vec3d &center);
}  // namespace vem
