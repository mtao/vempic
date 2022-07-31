#include "vem/mesh.hpp"
#include "vem/poisson_2d/constraints.hpp"

namespace vem::poisson_2d {
// constructs dirichlet boundary conditions for a linear function
ScalarConstraints linear_function_dirichlet(const VEMMesh2 &mesh,
                                            const double constant,
                                            const mtao::Vec2d &linear);

// constructs neumann boundary conditions for a linear function, with the mean
// value set to constant
ScalarConstraints linear_function_neumann(const VEMMesh2 &mesh,
                                          const double constant,
                                          const mtao::Vec2d &linear);

// constructs 0 dirichlet boundary conditions on the boundary and 0 on the
// membrane
ScalarConstraints pulled_membrane(const VEMMesh2 &mesh, const size_t index,
                                  const double value);

// constructs neumann boundary conditions for a linear function, with the mean
// value set to constant.
// currently is just evaluated by a single sample, but quadrature would be good
ScalarConstraints neumann_from_boundary_function(
    const VEMMesh2 &mesh,
    const std::function<std::tuple<mtao::Vec2d, bool>(const mtao::Vec2d,
                                                      double)> &f,
    double t = 0);
}  // namespace vem::poisson_2d
