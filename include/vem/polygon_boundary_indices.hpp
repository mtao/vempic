#pragma once
#include <map>
#include <mtao/geometry/mesh/edges_to_polygons.hpp>
#include <mtao/types.hpp>
#include <optional>
#include <set>
#include <vector>

namespace vem {


struct PolygonBoundaryIndices
    : public mtao::geometry::mesh::PolygonBoundaryIndices {
    using Base = mtao::geometry::mesh::PolygonBoundaryIndices;
    PolygonBoundaryIndices &operator=(const PolygonBoundaryIndices &) = default;
    PolygonBoundaryIndices &operator=(const Base &b) {
        static_cast<Base *>(this)->operator=(b);
        return *this;
    }
    PolygonBoundaryIndices() = default;
    PolygonBoundaryIndices(std::vector<int> b,
                           std::set<std::vector<int>> h = {})
        : Base(b, h) {}
    mtao::ColVecs3i triangulate(const mtao::ColVecs2d &V) const;
    bool is_inside(const mtao::ColVecs2d &V, const mtao::Vec2d &p) const;
};
}
