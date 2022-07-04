#include "vem/serialization/serialize_vdb.hpp"

#include <openvdb/Types.h>
#include <openvdb/math/BBox.h>
#include <openvdb/math/Coord.h>
#include <openvdb/openvdb.h>

#include "vem/monomial_field_embedder3.hpp"
namespace vem::serialization {

void serialize_scalar_field_with_vdb(Inventory& inventory,
                                     const std::string& name,
                                     const MonomialBasisIndexer3& indexer,
                                     const mtao::VecXd& coeffs,
                                     double voxel_size) {
    // other than a minor lock doing this each frame costs almost nothing
    openvdb::initialize();
    auto transform = openvdb::math::Transform::createLinearTransform(
        /*voxel size=*/voxel_size);
    openvdb::FloatGrid::Ptr grid =
        openvdb::FloatGrid::create(/*background value=*/0.0);

    grid->setTransform(transform);
    grid->setGridClass(openvdb::GRID_LEVEL_SET);

    auto bbox = indexer.mesh().bounding_box();

    openvdb::math::BBox<openvdb::Vec3d> vdb_bbox;
    {
        auto m = bbox.min();
        auto M = bbox.max();
        auto& vm = vdb_bbox.min();
        auto& vM = vdb_bbox.max();
        vm.x() = m.x();
        vm.y() = m.y();
        vm.z() = m.z();
        vM.x() = M.x();
        vM.y() = M.y();
        vM.z() = M.z();
    }
    openvdb::math::CoordBBox cbbox;
    cbbox.min() = openvdb::math::Coord(
        openvdb::math::Vec3i(grid->worldToIndex(vdb_bbox.min())));
    cbbox.max() = openvdb::math::Coord(
        openvdb::math::Vec3i(grid->worldToIndex(vdb_bbox.max())));
    cbbox.max().x()++;
    cbbox.max().y()++;
    cbbox.max().z()++;

    auto func = vem::MonomialScalarFieldEmbedder3(indexer);
    func.coefficients() = coeffs;
    // Get a voxel accessor.
    auto accessor = grid->getAccessor();
    // Compute the signed distance from the surface of the sphere of each
    // voxel within the bounding box and insert the value into the grid
    // if it is smaller in magnitude than the background value.
    openvdb::Coord ijk;
    int &i = ijk[0], &j = ijk[1], &k = ijk[2];
    for (i = cbbox.min()[0]; i < cbbox.max()[0]; ++i) {
        for (j = cbbox.min()[1]; j < cbbox.max()[1]; ++j) {
            for (k = cbbox.min()[2]; k < cbbox.max()[2]; ++k) {
                auto p = grid->indexToWorld(ijk);
                accessor.setValue(
                    ijk, func.get_value(mtao::Vec3d(p[0], p[1], p[2])));
            }
        }
    }
    grid->setName("values");
    auto p = inventory.get_new_asset_path(name, "vdb");
    openvdb::io::File(std::string(p)).write({grid});
}

void serialize_vector_field_with_vdb(Inventory& inventory,
                                     const std::string& name,
                                     const MonomialBasisIndexer3& indexer,
                                     const mtao::ColVecs3d& coeffs,
                                     double voxel_size) {
    // other than a minor lock doing this each frame costs almost nothing
    openvdb::initialize();
    auto transform = openvdb::math::Transform::createLinearTransform(
        /*voxel size=*/voxel_size);
    openvdb::Vec3dGrid::Ptr grid = openvdb::Vec3dGrid::create(
        /*background value=*/openvdb::Vec3d(0.0, 0.0, 0.0));

    grid->setTransform(transform);
    grid->setGridClass(openvdb::GRID_UNKNOWN);

    auto bbox = indexer.mesh().bounding_box();

    openvdb::math::BBox<openvdb::Vec3d> vdb_bbox;
    {
        auto m = bbox.min();
        auto M = bbox.max();
        auto& vm = vdb_bbox.min();
        auto& vM = vdb_bbox.max();
        vm.x() = m.x();
        vm.y() = m.y();
        vm.z() = m.z();
        vM.x() = M.x();
        vM.y() = M.y();
        vM.z() = M.z();
    }
    openvdb::math::CoordBBox cbbox;
    cbbox.min() = openvdb::math::Coord(
        openvdb::math::Vec3i(grid->worldToIndex(vdb_bbox.min())));
    cbbox.max() = openvdb::math::Coord(
        openvdb::math::Vec3i(grid->worldToIndex(vdb_bbox.max())));
    cbbox.max().x()++;
    cbbox.max().y()++;
    cbbox.max().z()++;

    auto func = vem::MonomialVectorFieldEmbedder3(indexer);
    func.coefficients() = coeffs;
    // Get a voxel accessor.
    auto accessor = grid->getAccessor();
    // Compute the signed distance from the surface of the sphere of each
    // voxel within the bounding box and insert the value into the grid
    // if it is smaller in magnitude than the background value.
    openvdb::Coord ijk;
    int &i = ijk[0], &j = ijk[1], &k = ijk[2];
    for (i = cbbox.min()[0]; i < cbbox.max()[0]; ++i) {
        for (j = cbbox.min()[1]; j < cbbox.max()[1]; ++j) {
            for (k = cbbox.min()[2]; k < cbbox.max()[2]; ++k) {
                auto p = grid->indexToWorld(ijk);
                auto v = func.get_vector(mtao::Vec3d(p[0], p[1], p[2]));
                accessor.setValue(ijk,
                                  openvdb::math::Vec3d(v.x(), v.y(), v.z()));
            }
        }
    }
    grid->setName("vectors");
    auto p = inventory.get_new_asset_path(name, "vdb");
    openvdb::io::File(std::string(p)).write({grid});
}
}  // namespace vem::serialization
