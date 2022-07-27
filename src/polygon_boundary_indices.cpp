#include "vem/polygon_boundary_indices.hpp"
#include <mtao/geometry/mesh/triangle/triangle_wrapper.h>
#include <mtao/geometry/mesh/earclipping.hpp>

namespace vem {
bool PolygonBoundaryIndices::is_inside(const mtao::ColVecs2d &V,
                                       const mtao::Vec2d &p) const {
    double wn = mtao::geometry::winding_number(V, *this, p);
    for (auto &&h : holes) {
        wn += mtao::geometry::winding_number(V, h, p);
    }

    return std::abs(wn) > .5;
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
}

