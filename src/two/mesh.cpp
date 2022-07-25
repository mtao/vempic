#include "vem/mesh.hpp"

#include <mtao/geometry/mesh/triangle/triangle_wrapper.h>

#include <mtao/geometry/mesh/earclipping.hpp>
#include <mtao/geometry/mesh/edges_to_polygons.hpp>
#include <mtao/geometry/volume.hpp>
#include <numeric>

#include "vem/edge_lengths.hpp"
namespace vem {

bool PolygonBoundaryIndices::is_inside(const mtao::ColVecs2d &V,
                                       const mtao::Vec2d &p) const {
    double wn = mtao::geometry::winding_number(V, *this, p);
    for (auto &&h : holes) {
        wn += mtao::geometry::winding_number(V, h, p);
    }

    return std::abs(wn) > .5;
}

Eigen::AlignedBox<double, 2> VEMMesh2::bounding_box() const {
    return Eigen::AlignedBox<double, 2>(V.rowwise().minCoeff(),
                                        V.rowwise().maxCoeff());
}
std::set<int> VEMMesh2::boundary_edge_indices() const { return {}; }
std::vector<std::set<int>> VEMMesh2::cell_regions() const {
    return {};
    // currently all of the code assumes that empty set = all, so this following
    // bit isn't necessary

    // std::vector<std::set<int>> a(1);
    // for (int j = 0; j < cell_count(); ++j) {
    //    a[0].emplace(j);
    //}
    // return a;
}
double VEMMesh2::dx() const { return edge_lengths(*this).mean(); }

std::vector<mtao::ColVecs3i> VEMMesh2::triangulated_faces() const {
    std::vector<mtao::ColVecs3i> faces;
    auto boundary_loops = face_loops();
    faces.reserve(boundary_loops.size());
    std::transform(boundary_loops.begin(), boundary_loops.end(),
                   std::back_inserter(faces),
                   [&](auto &&loop) { return loop.triangulate(V); });
    return faces;
}
auto VEMMesh2::face_loops() const -> std::vector<PolygonBoundaryIndices> {
    std::vector<PolygonBoundaryIndices> ret(cell_count());

    std::transform(face_boundary_map.begin(), face_boundary_map.end(),
                   ret.begin(), [&](const std::map<int, bool> &map) {
                       mtao::ColVecs2i E(2, map.size());
                       for (auto &&[idx, pr] : mtao::iterator::enumerate(map)) {
                           auto e = E.col(idx);
                           auto me = this->E.col(pr.first);
                           if (pr.second) {
                               e = me.reverse();
                           } else {
                               e = me;
                           }
                       }
                       auto loops =
                           mtao::geometry::mesh::edges_to_polygons(V, E);

                       auto it = loops.begin();
                       double vol = std::abs(mtao::geometry::curve_volume(
                           V, it->begin(), it->end()));
                       for (auto it2 = it + 1; it2 != loops.end(); ++it2) {
                           double nvol = std::abs(mtao::geometry::curve_volume(
                               V, it2->begin(), it2->end()));
                           if (nvol > vol) {
                               it = it2;
                               vol = nvol;
                           }
                       }
                       return *it;
                   });
    return ret;
}

mtao::ColVecs3i PolygonBoundaryIndices::triangulate(
    const mtao::ColVecs2d &V) const {
    if (Base::empty()) {
        return {};
    }
    if (holes.size() == 0) {
        // spdlog::info("Cell: {}", fmt::join(*this,","));
        auto F = mtao::geometry::mesh::earclipping(V, *this);
        // std::cout << "Earclipping result:\n" << F << std::endl;
        return F;
    } else {
        constexpr static bool do_add_vertices = false;
        // collect the number of edges
        int size = this->size();
        for (auto &&v : holes) {
            size += v.size();
        }

        // check the total number of vertices
        std::set<int> used_vertices;
        for (auto &&c : *this) {
            used_vertices.insert(c);
        }
        for (auto &&v : holes) {
            for (auto &&c : v) {
                used_vertices.insert(c);
            }
        }
        if (used_vertices.size() == 0) {
            return {};
        }

        // compactify the vertices
        mtao::map<int, int> reindexer;
        mtao::map<int, int> unreindexer;
        mtao::ColVecs2d newV(2, used_vertices.size());
        for (auto &&[i, v] : mtao::iterator::enumerate(used_vertices)) {
            reindexer[v] = i;
            newV.col(i) = V.col(v);
            unreindexer[i] = v;
        }

        // compactify the edges
        mtao::ColVecs2i E(2, size);
        mtao::vector<mtao::Vec2d> holes;
        size = 0;
        for (int i = 0; i < this->size(); ++i) {
            auto e = E.col(size++);
            e(0) = reindexer[(*this)[i]];
            e(1) = reindexer[(*this)[(i + 1) % this->size()]];
        }
        for (auto &&curve : holes) {
            for (int i = 0; i < this->size(); ++i) {
                auto e = E.col(size++);
                e(0) = reindexer[curve[i]];
                e(1) = reindexer[curve[(i + 1) % this->size()]];
            }
        }
        mtao::geometry::mesh::triangle::Mesh m(newV, E);

        // set some arbitrary attributes?
        m.fill_attributes();
        for (int i = 0; i < m.EA.cols(); ++i) {
            m.EA(i) = i;
        }
        for (int i = 0; i < m.VA.cols(); ++i) {
            m.VA(i) = i;
        }
        bool points_added = false;
        mtao::ColVecs2d newV2;
        mtao::ColVecs3i newF;

        if (do_add_vertices) {
            // static const std::string str ="zPa.01qepcDQ";
            static const std::string str = "pcePzQYY";
            std::cerr << "I bet im about to crash" << std::endl;
            auto nm = mtao::geometry::mesh::triangle::triangle_wrapper(
                m, std::string_view(str));
            std::cerr << "I lost a bet" << std::endl;
            if (nm.V.cols() > newV.cols()) {
                newV2 = nm.V;
                points_added = true;
            }

            newF = nm.F;
        } else {
            static const std::string str = "pcePzQYY";
            auto nm = mtao::geometry::mesh::triangle::triangle_wrapper(
                m, std::string_view(str));
            newF = nm.F;
        }
        for (int i = 0; i < newF.size(); ++i) {
            auto &&f = newF(i);
            if (unreindexer.find(f) == unreindexer.end()) {
                points_added = true;
                break;
            }
        }
        if (!points_added) {
            for (int i = 0; i < newF.size(); ++i) {
                auto &&f = newF(i);
                f = unreindexer[f];
            }
        }

        std::set<int> interior;
        for (int i = 0; i < newF.cols(); ++i) {
            mtao::Vec2d B = mtao::Vec2d::Zero();
            auto f = newF.col(i);
            for (int j = 0; j < 3; ++j) {
                if (points_added) {
                    B += newV2.col(f(j));
                } else {
                    B += V.col(f(j));
                }
            }
            B /= 3;
            double wn = mtao::geometry::winding_number(V, *this, B);
            for (auto &&c : holes) {
                double mywn = mtao::geometry::winding_number(V, c, B);
                wn += mywn;
            }
            // std::cout << wn << " ";
            if (std::abs(wn) > .5) {
                interior.insert(i);
            }
        }
        // std::cout << std::endl;
        if (interior.size() == 0) {
            for (int i = 0; i < newF.cols(); ++i) {
                interior.insert(i);
            }
        }
        mtao::ColVecs3i FF(3, interior.size());
        for (auto &&[i, b] : mtao::iterator::enumerate(interior)) {
            FF.col(i) = newF.col(b);
        }
        // std::cout << std::endl;
        if (points_added && !do_add_vertices) {
            std::cout << "points were added when they shouldnt have!"
                      << std::endl;
            return {};
        }
        return FF;
    }
}

double VEMMesh2::diameter(size_t cell_index) const {
    if (debug_diameter) {
        return 1;
    }
    // std::cout << cell_count() << std::endl;
    // std::cout << E << std::endl;
    // std::cout << V.cols() << std::endl;
    // first compute the radii - i.e the furthest distance from teh center
    const auto &fbm = face_boundary_map.at(cell_index);
    std::set<size_t> verts;
    for (auto &&[eidx, sgn] : fbm) {
        auto e = E.col(eidx);
        verts.emplace(e(0));
        verts.emplace(e(1));
    }
    double d = 0;
    auto c = C.col(cell_index);
    // we're doing things twice, which is stupid. but who cares
    for (auto &&v : verts) {
        for (auto &&u : verts) {
            d = std::max(d, (V.col(v) - V.col(u)).norm());
        }
    }
    return d;
}
double VEMMesh2::boundary_diameter(size_t face_index) const {
    auto e = E.col(face_index);
    auto a = V.col(e(0));
    auto b = V.col(e(1));
    return (a - b).norm();
}
mtao::Vec2d VEMMesh2::boundary_center(size_t edge_index) const {
    auto e = E.col(edge_index);
    auto a = V.col(e(0));
    auto b = V.col(e(1));
    return (a + b) / 2.0;
}

}  // namespace vem
