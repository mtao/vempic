#pragma once
#include <Eigen/Core>
#include <mtao/types.hpp>
#include <set>
#include <vector>

#include "vem/mesh.hpp"

namespace vem::utils {
template <typename MeshType>
class CellIdentifier {
   public:
    template <typename Derived>
    int get_cell(const Eigen::MatrixBase<Derived> &p,
                 int last_known = -1) const;
    template <typename Derived>
    mtao::VecXi get_cells(const Eigen::MatrixBase<Derived> &p,
                          const mtao::VecXi &last_known = {}) const;

    template <typename Derived>
    std::vector<std::set<int>> cell_ownerships(
        const Eigen::MatrixBase<Derived> &P,
        const mtao::VecXi &last_known = {}) const;
    CellIdentifier(const MeshType&m) : _mesh(m) {}
    CellIdentifier(const CellIdentifier &) = default;
    CellIdentifier(CellIdentifier &&) = default;

   private:
    const MeshType &_mesh;
};
template <typename MeshType>
template <typename Derived>
int CellIdentifier<MeshType>::get_cell(const Eigen::MatrixBase<Derived> &p,
                                       int last_known) const {
    return _mesh.get_cell(p, last_known);
}

template <typename MeshType>
template <typename Derived>
mtao::VecXi CellIdentifier<MeshType>::get_cells(
    const Eigen::MatrixBase<Derived> &p, const mtao::VecXi &i) const {
    mtao::VecXi d(p.cols());
    const bool have_last_known = i.size() != 0;
    for (int j = 0; j < p.size(); ++j) {
        d(j) = get_cell(p.col(j), have_last_known ? i(j) : -1);
    }
    return d;
}
template <typename MeshType>
template <typename Derived>
std::vector<std::set<int>> CellIdentifier<MeshType>::cell_ownerships(
    const Eigen::MatrixBase<Derived> &P, const mtao::VecXi &last_known) const {
    auto a = get_cells(P, last_known);
    std::vector<std::set<int>> ret(_mesh.cell_count());
    for (int point_index = 0; point_index < a.size(); ++point_index) {
        int cell_index = a(point_index);
        if (cell_index >= 0 && cell_index < _mesh.cell_count()) {
            ret[cell_index].emplace(point_index);
        }
    }
    return ret;
}

}  // namespace vem::utils
