#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mtao/types.hpp>

#include "vem/serialization/inventory.hpp"
namespace mtao {
    using ColVecs5d = ColVectors<double,5>;
    using ColVecs6d = ColVectors<double,6>;

}
namespace vem::serialization {

void serialize_sparse_matrix(Inventory& inventory, const std::string& name,
                             const Eigen::SparseMatrix<double>& A);
void serialize_VecXd(Inventory& inventory, const std::string& name,
                     const mtao::VecXd& A);
void serialize_obj(Inventory& inventory, const std::string& name,
                   const mtao::ColVecs3d& V, const mtao::ColVecs3i& F);

void serialize_points6(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs6d& P);
void serialize_points5(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs5d& P);
void serialize_points4(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs4d& P);
void serialize_points3(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs3d& P);
void serialize_points2(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs2d& P);

Eigen::SparseMatrix<double> deserialize_sparse_matrix(
    const Inventory& inventory, const std::string& name);
mtao::VecXd deserialize_VecXd(const Inventory& inventory,
                              const std::string& name);

mtao::VecXi deserialize_VecXi(const Inventory& inventory,
                              const std::string& name);

std::tuple<mtao::ColVecs3d, mtao::ColVecs3i> deserialize_obj3(
    const Inventory& inventory, const std::string& name);
mtao::ColVecs3d deserialize_points3(const Inventory& inventory,
                                    const std::string& name);

mtao::ColVecs2d deserialize_points2(const Inventory& inventory,
                                    const std::string& name);
mtao::ColVecs4d deserialize_points4(const Inventory& inventory,
                                    const std::string& name);
mtao::ColVecs5d deserialize_points5(const Inventory& inventory,
                                    const std::string& name);

mtao::ColVecs6d deserialize_points6(const Inventory& inventory,
                                    const std::string& name);
}  // namespace vem::serialization
