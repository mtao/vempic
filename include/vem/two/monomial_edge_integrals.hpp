#include <map>
#include <mtao/types.hpp>

#include "mesh.hpp"

namespace vem::two {

mtao::VecXd scaled_monomial_edge_integrals(const VEMMesh2 &mesh, int index, double scale, int max_degree);
mtao::VecXd scaled_monomial_edge_integrals(const VEMMesh2 &mesh, int index, double scale, int max_degree, const mtao::Vec2d &center);

std::map<int, std::vector<double>> per_edge_scaled_monomial_edge_integrals(
  const VEMMesh2 &mesh,
  int index,
  double scale,
  int max_degree);
std::map<int, std::vector<double>> per_edge_scaled_monomial_edge_integrals(
  const VEMMesh2 &mesh,
  int index,
  double scale,
  int max_degree,
  const mtao::Vec2d &center);

std::vector<double> single_edge_scaled_monomial_edge_integrals(
  const VEMMesh2 &mesh,
  int cell_index,
  int edge_index,
  double scale,
  int max_degree);
std::vector<double> single_edge_scaled_monomial_edge_integrals(
  const VEMMesh2 &mesh,
  int cell_index,
  int edge_index,
  double scale,
  int max_degree,
  const mtao::Vec2d &center);
}// namespace vem
