
#include "vem/utils/in_triangle_mesh.hpp"

#include <igl/WindingNumberAABB.h>
#include <igl/fast_winding_number.h>
namespace vem::utils {

struct InTriangleMesh::_Impl {
    _Impl(Eigen::MatrixXd V, Eigen::MatrixXi F)
    {
        igl::fast_winding_number(V,F,2,fwn_bvh);
        }

    bool operator()(const mtao::Vec3d& p) const {
        mtao::VecXd W;
        igl::fast_winding_number(fwn_bvh,2,p.transpose(),W);
        return std::abs(W(0)) > .5;
    }
    
    igl::FastWindingNumberBVH fwn_bvh;
};
bool InTriangleMesh::operator()(const mtao::Vec3d& p) const { return (*_D)(p); }
InTriangleMesh::InTriangleMesh(Eigen::MatrixXd V, Eigen::MatrixXi F)
    : _D(std::make_unique<_Impl>(std::move(V), std::move(F))) {}
InTriangleMesh::~InTriangleMesh() {}
}  // namespace vem::utils
