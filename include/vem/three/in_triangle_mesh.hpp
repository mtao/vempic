#pragma once
#include <memory>
#include <mtao/types.hpp>

#include "vem/utils/inside_geometry_predicate.hpp"
namespace vem::three {
class InTriangleMesh : public utils::InsideGeometryPredicate {
   public:
    bool operator()(const mtao::Vec3d& p) const;
    InTriangleMesh(Eigen::MatrixXd V, Eigen::MatrixXi F);
    ~InTriangleMesh();

   private:
    struct _Impl;
    std::unique_ptr<_Impl> _D;
};
};  // namespace vem::utils

