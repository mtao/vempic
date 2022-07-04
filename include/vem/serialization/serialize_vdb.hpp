#pragma once

#include "vem/monomial_basis_indexer.hpp"
#include "vem/serialization/inventory.hpp"
namespace vem::serialization {

void serialize_scalar_field_with_vdb(Inventory& inventory,
                                     const std::string& name,
                                     const MonomialBasisIndexer3& indexer,
                                     const mtao::VecXd& coeffs,
                                     double voxel_size);
void serialize_vector_field_with_vdb(Inventory& inventory,
                                     const std::string& name,
                                     const MonomialBasisIndexer3& indexer,
                                     const mtao::ColVecs3d& coeffs,
                                     double voxel_size);
}  // namespace vem::serialization
