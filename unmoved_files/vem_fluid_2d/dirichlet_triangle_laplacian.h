#pragma once
#include <map>

#include <mtao/types.hpp>
mtao::VecXd dirichlet_laplacian(const mtao::ColVecs2d& V, const mtao::ColVecs3i& F, const std::map<size_t,double>& dv);
