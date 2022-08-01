
#include <mtao/geometry/mesh/triangle/triangle_wrapper.h>
#include <tbb/parallel_for.h>

#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/mesh/earclipping.hpp>
#include <mtao/geometry/mesh/edges_to_polygons.hpp>
#if defined(_OPENMP)
#undef _OPENMP
#include <mtao/geometry/volume.hpp>
#define _OPENMP
#endif
#include <numeric>

#include "vem/three/mesh.hpp"
#include "vem/three/boundary_facets.hpp"
#include "vem/three/cell_boundary_facets.hpp"
#include "vem/three/face_boundary_facets.hpp"
namespace vem::three {
VEMMesh3::~VEMMesh3() {}

int VEMMesh3::grade(size_t cell_index) const { return 0; }
std::optional<int> VEMMesh3::cell_category(size_t cell_index) const {
    return {};
}
Eigen::AlignedBox<double, 3> VEMMesh3::bounding_box() const {
    return Eigen::AlignedBox<double, 3>(V.rowwise().minCoeff(),
                                        V.rowwise().maxCoeff());
}
std::vector<std::set<int>> VEMMesh3::cell_regions() const {
    return {};
    // currently all of the code assumes that empty set = all, so this following
    // bit isn't necessary

    // std::vector<std::set<int>> a(1);
    // for (int j = 0; j < cell_count(); ++j) {
    //    a[0].emplace(j);
    //}
    // return a;
}

mtao::VecXi VEMMesh3::get_cells(const mtao::ColVecs3d& P,
                                const mtao::VecXi& last_known) const {
    bool has_last_known = last_known.size() == P.cols();
    mtao::VecXi I(P.cols());
    tbb::parallel_for<int>(0, P.cols(), [&](int j) {
        if (has_last_known) {
            I(j) = get_cell(P.col(j), last_known(j));
        } else {
            I(j) = get_cell(P.col(j));
        }
    });
    return I;
}
double VEMMesh3::diameter(size_t cell_index) const {
    // std::cout << cell_count() << std::endl;
    // std::cout << E << std::endl;
    // std::cout << V.cols() << std::endl;
    // first compute the radii - i.e the furthest distance from teh center
    auto verts = cell_boundary_vertices(*this, cell_index);
    double r = 0;
    auto c = C.col(cell_index);
    // we're doing things twice, which is stupid. but who cares
    for (auto&& v : verts) {
        if(v < V.cols()) {
        r = std::max(r, (V.col(v) - c).norm());
        }
    }
    return 2 * r;
}

double VEMMesh3::face_diameter(size_t face_index) const {
    // std::cout << cell_count() << std::endl;
    // std::cout << E << std::endl;
    // std::cout << V.cols() << std::endl;
    // first compute the radii - i.e the furthest distance from teh center
    auto verts = face_boundary_vertices(*this, face_index);
    double r = 0;
    auto c = FC.col(face_index);
    // we're doing things twice, which is stupid. but who cares
    for (auto&& v : verts) {
        r = std::max(r, (V.col(v) - c).norm());
    }
    return 2 * r;
}

double VEMMesh3::surface_area(int face_index) const {
    const auto& F = triangulated_faces.at(face_index);
    auto vol = mtao::geometry::volumes(V, F);
    return vol.sum();
}

mtao::Vec4d VEMMesh3::plane_equation(int face_index) const {
    mtao::Vec4d r;
    r.head<3>() = normal(face_index);
    const auto& F = triangulated_faces.at(face_index);
    r(3) = -r.head<3>().dot(V.col(F(0)));
    return r;
}
std::tuple<mtao::ColVecs3d, mtao::ColVecs3i, std::map<size_t, std::set<size_t>>>
VEMMesh3::collision_mesh(const std::set<int>& active_cells) const {
    auto bfm = boundary_face_map(*this, active_cells);
    mtao::vector<mtao::ColVecs3i> Fs;
    Fs.reserve(bfm.size());
    std::map<size_t, std::set<size_t>> mp;
    spdlog::info("Constructing collision mesh, got {} faces", bfm.size());
    for (auto&& [fidx, c] : bfm) {
        auto F = triangulated_faces.at(fidx);
        bool sgn = cell_boundary_map.at(c).at(fidx);
        if (!sgn) {
            auto tmp = F.row(1).eval();
            F.row(1) = F.row(2);
            F.row(2) = tmp;
        }
        Fs.emplace_back(std::move(F));

        for (int j = 0; j < F.cols(); ++j) {
            mp[mp.size()].emplace(fidx);
        }
    }

    auto F = mtao::eigen::hstack_iter(Fs.begin(), Fs.end());
    return {V, F, mp};
}
}  // namespace vem
