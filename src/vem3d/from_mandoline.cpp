#include "vem/from_mandoline.hpp"

#include <mandoline/construction/face_collapser.hpp>
#include <mandoline/operators/region_boundaries2.hpp>
#include <mandoline/operators/volume2.hpp>
#include <mtao/geometry/volume.hpp>

#include "mandoline/construction/construct2.hpp"
#include "vem/utils/merge_cells.hpp"
namespace vem {

double MandolineVEMMesh2::dx() const {
    auto dx = _ccm.dx();
    return *std::min_element(dx.begin(), dx.end());
}
std::vector<std::set<int>> MandolineVEMMesh2::cell_regions() const {
    auto R = _ccm.regions();

    spdlog::info("R Range: {} {}", R.minCoeff(), R.maxCoeff());
    std::vector<std::set<int>> regions(R.maxCoeff() + 1);
    for (int j = 0; j < R.size(); ++j) {
#if defined(MERGE_2D_CELLS)
        spdlog::info("reg[{}] / {}, ccmc{} / {}", R(j), regions.size(), j,
                     _ccm_cell_to_cell.size());
        regions.at(R(j)).emplace(_ccm_cell_to_cell.at(j));
#else
        regions.at(R(j)).emplace(j);
#endif
        // std::cout << _ccm_cell_to_cell[j] << " " << j << " / "
        //          << _ccm_cell_to_cell.size() << std::endl;
        // regions[R(j)].emplace(j);
    }

    return regions;
}
auto MandolineVEMMesh2::face_loops() const
    -> std::vector<PolygonBoundaryIndices> {
    std::vector<PolygonBoundaryIndices> ret(_ccm.num_cells());
    auto V = _ccm.vertices();
    spdlog::info("Doing mandolien specfic face loop");

    for (int j = 0; j < cell_count(); ++j) {
#if defined(MERGE_2D_CELLS)
        for (auto &&cidx : _cell_to_ccm_cells[j]) {
#else
        int cidx = j;
#endif
    spdlog::info("Doing loop {}", cidx);
            auto lset = _ccm.cell(cidx);
            if (lset.size() == 0) {
                spdlog::warn("No loops in cell {}", cidx);
                continue;
            } else if (lset.size() == 1) {
                spdlog::info("Single loop in cell {}", cidx);
                auto &pbi = ret[j] = PolygonBoundaryIndices(*lset.begin());
            } else {
                spdlog::info("Cell with holes in cell {}", cidx);
                std::vector<std::vector<int>> ll(lset.begin(), lset.end());
                std::vector<double> vols(ll.size());
                std::transform(ll.begin(), ll.end(), vols.begin(),
                               [&](auto &&loop) {
                                   return mtao::geometry::curve_volume(V, loop);
                               });
                size_t dist = std::distance(
                    vols.begin(), std::max_element(vols.begin(), vols.end()));
                auto it = ll.begin() + dist;
                auto &pbi = ret[j] = PolygonBoundaryIndices(std::move(*it));
                ll.erase(it);
                for (auto &&l : ll) {
                    pbi.holes.emplace(std::move(l));
                }
            }
#if defined(MERGE_2D_CELLS)
        }
#endif
    }
    return ret;
}
MandolineVEMMesh2::MandolineVEMMesh2(
    const mandoline::CutCellMesh<2> &ccm,
    const std::optional<double> &merge_cell_volume, bool delaminate_boundaries)
    : _ccm(ccm) {
    this->E = ccm.edges();
    this->V = ccm.vertices();
    this->face_boundary_map.resize(ccm.num_cells());
    const auto &ccmB = ccm.m_face_boundary_map;
    spdlog::warn("CCM face boundary map size: {}", ccmB.size());

    // add the boundary facets for the nontrivial cut-cells
    for (auto &&[cidx, mp] : ccmB) {
        auto &a = this->face_boundary_map[cidx];
        std::transform(mp.begin(), mp.end(), std::inserter(a, a.end()),
                       [](const std::pair<const int, bool> &pr)
                           -> std::pair<const size_t, bool> {
                           auto [a, b] = pr;
                           return {size_t(a), b};
                       });
    }
    // add the boundaries for the boundary between trivial and nontrivial cells
    for (auto &&[eidx, edge] : mtao::iterator::enumerate(ccm.cut_edges())) {
        if (edge.external_boundary) {
            auto [bound, sgn] = *edge.external_boundary;
            if (bound >= 0) {
                auto e = E.col(eidx);
                auto u = V.col(e(1));
                auto v = V.col(e(0));
                auto c = _ccm.cell_grid().vertex(bound);
                bool flip =
                    (u - c).homogeneous().cross((v - c).homogeneous()).z() > 0;
                // auto cell = ccm.cell_grid().unindex(bound);
                int cell_index = ccm.exterior_grid.cell_indices().get(bound) +
                                 ccm.cut_faces().size();
                // this->face_boundary_map[cell_index][eidx] = sgn ? -1 : 1;
                this->face_boundary_map[cell_index][eidx] = flip;
                // spdlog::info(
                //    "External boundary face: {} to cell {} with sign {}",
                //    eidx, cell_index, sgn);
            }
        }
    }
    std::map<int, std::array<int, 2>> ccm_exterior_grid_unindexer;
    for (auto &&[idx, cell] :
         mtao::iterator::enumerate(_ccm.exterior_grid.cell_coords())) {
        ccm_exterior_grid_unindexer[idx] = cell;
    }

    // add non-trivial cells
    for (auto &&[eidx_noff, pr] :
         mtao::iterator::enumerate(ccm.exterior_grid.boundary_facet_pairs())) {
        int eidx = eidx_noff + ccm.num_cutedges();
        int axis = ccm.exterior_grid.get_face_axis(eidx_noff);
        bool flip = axis == 1;
        auto [a, b] = pr;
        auto e = E.col(eidx);
        auto u = V.col(e(1));
        auto v = V.col(e(0));

        if (a >= 0) {
            auto c = _ccm.cell_grid().vertex(ccm_exterior_grid_unindexer.at(a));
            bool flip =
                (u - c).homogeneous().cross((v - c).homogeneous()).z() > 0;
            this->face_boundary_map[a + ccm.num_cutfaces()][eidx] = flip;
            // false ^ flip;
        }
        if (b >= 0) {
            auto c = _ccm.cell_grid().vertex(ccm_exterior_grid_unindexer.at(b));
            bool flip =
                (u - c).homogeneous().cross((v - c).homogeneous()).z() > 0;
            this->face_boundary_map[b + ccm.num_cutfaces()][eidx] =
                flip;  // true ^ flip;
        }
    }

    this->C.resize(2, ccm.num_cells());
    // make the centers the grid cell centers
    for (auto &&[cidx, ccindx] : ccm.cell_grid_ownership) {
        auto center = ccm.cell_grid().vertex(cidx);
        for (auto &&c : ccindx) {
            this->C.col(c) = center;
        }
    }

    for (auto &&[ind, cell] :
         mtao::iterator::enumerate(ccm.exterior_grid.cell_coords())) {
        this->C.col(ind + ccm.num_cutcells()) = ccm.cell_grid().vertex(cell);
    }
    /*
    mtao::VecXd B = _ccm.boundary(true) * mtao::VecXd::Ones(_ccm.cell_size());
    for (int j = 0; j < B.size(); ++j) {
        if (B(j) != 0) {
            std::cout << j << " ==> " << E.col(j) << std::endl;
        }
    }

    std::cout << B.transpose() << std::endl;
    B.setZero();
    for (auto&& c : face_boundary_map) {
        for (auto&& [f, s] : c) {
            B(f) += (s ? -1 : 1);
        }
    }
    std::cout << B.transpose() << std::endl;
    */

    neighboring_cells.resize(cell_count());
    ccm.cell_grid().loop_parallel([&](const std::array<int, 2> &a) {
        int cell_index = ccm.StaggeredGrid::cell_index(a);
        auto &pool = neighboring_cells.at(cell_index);
        std::array<int, 2> min = a;
        std::array<int, 2> max = a;
        for (auto &&v : min) {
            --v;
        }
        for (auto &&v : max) {
            v += 2;
        }
        mtao::geometry::grid::utils::multi_loop(
            min, max, [&](const std::array<int, 2> &c) {
                if (ccm.StaggeredGrid::cell_grid().valid_index(c)) {
                    int cell_index = ccm.StaggeredGrid::cell_index(c);
                    pool.emplace(cell_index);
                }
            });
    });

    if (delaminate_boundaries) {
        std::set<int> BEI;
        for (auto &&[eidx, e] : ccm.mesh_cut_edges()) {
            BEI.emplace(eidx);
        }

        std::set<int> seen_edges;
        std::set<int> to_delaminate;

        // first pass to find which faces do need to be delaminated
        for (auto &&c : face_boundary_map) {
            for (auto &&[eidx, sgn] : c) {
                if (BEI.contains(eidx)) {
                    auto [it, added] = seen_edges.emplace(eidx);
                    if (!added) {
                        to_delaminate.emplace(eidx);
                    }
                }
            }
        }

        int cur_pos = E.cols();
        E.conservativeResize(2, E.cols() + to_delaminate.size());

        for (auto &&c : face_boundary_map) {
            bool changes = false;
            for (auto &&[eidx, sgn] : c) {
                if (sgn && to_delaminate.contains(eidx)) {
                    changes = true;
                    break;
                }
            }
            if (changes) {
                std::map<int, bool> new_c = c;
                for (auto &&[eidx, sgn] : c) {
                    if (sgn && to_delaminate.contains(eidx)) {
                        int index = cur_pos++;
                        E.col(index) = E.col(eidx);
                        new_c.erase(eidx);
                        new_c[index] = sgn;
                    }
                }
                c = std::move(new_c);
            }
        }
    }

    // for (auto &&[idx, c] : mtao::iterator::enumerate(face_boundary_map)) {
    //    std::cout << idx << ")))";
    //    for (auto &&[eidx, sgn] : c) {
    //        std::cout << (sgn ? '-' : '+') << eidx << " ";
    //    }
    //    std::cout << std::endl;
    //}

#if defined(MERGE_2D_CELLS)
    if (merge_cell_volume) {
        double frac = std::clamp(*merge_cell_volume, 0., 1.);
        double threshold = _ccm.dx().prod() * frac;
        auto vol = mandoline::operators::face_volumes(_ccm, false);

        auto [new_cells, old_to_new] = vem::utils::merge_cells(
            face_boundary_map, vol, _ccm.regions(), threshold);

        _ccm_cell_to_cell = std::move(old_to_new);

        mtao::ColVecs2d CNew(2, new_cells.size());
        CNew.setZero();
        _cell_to_ccm_cells.resize(new_cells.size());
        for (auto &&[old, nu] : mtao::iterator::enumerate(old_to_new)) {
            _cell_to_ccm_cells[nu].emplace(old);
        }

        std::vector<std::set<int>> new_neighboring_cells(new_cells.size());
        for (auto &&[idx, old_cells, nbrs] : mtao::iterator::enumerate(
                 _cell_to_ccm_cells, new_neighboring_cells)) {
            auto c = CNew.col(idx);
            double accum = 0;
            for (auto &&cidx : old_cells) {
                double v = vol(cidx);
                auto oc = C.col(cidx);
                c += v * oc;
                accum += v;

                nbrs.emplace(_ccm_cell_to_cell[cidx]);
            }
            if (accum != 0) {
                c /= accum;
            }
        }

        spdlog::info("Merged {} cells to {} cells", face_boundary_map.size(),
                     new_cells.size());
        face_boundary_map = std::move(new_cells);
        neighboring_cells = std::move(new_neighboring_cells);

        // for (auto &&[idx, c] : mtao::iterator::enumerate(face_boundary_map))
        // {
        //    std::cout << idx << ")))";
        //    for (auto &&[eidx, sgn] : c) {
        //        std::cout << (sgn ? '-' : '+') << eidx << " ";
        //    }
        //    std::cout << std::endl;
        //}
    } else {
        _cell_to_ccm_cells.resize(face_boundary_map.size());
        for (auto &&[i, a] : mtao::iterator::enumerate(_ccm_cell_to_cell)) {
            _cell_to_ccm_cells[a].emplace(i);
        }
    }
#endif
}
bool MandolineVEMMesh2::in_cell(const mtao::Vec2d &p, int cell_index) const {
#if defined(MERGE_2D_CELLS)
    for (auto &&cidx : _cell_to_ccm_cells[cell_index]) {
        if (_ccm.in_cell(p, cidx)) {
            return true;
        }
    }
    return false;
#else
    return _ccm.in_cell(p, cell_index);
#endif
}

int MandolineVEMMesh2::get_cell(const mtao::Vec2d &p, int last_known) const {
#if defined(MERGE_2D_CELLS)
    return _ccm_cell_to_cell[_ccm.cell_index(p)];
#else
    return _ccm.cell_index(p);
#endif
}
MandolineVEMMesh2 from_mandoline(const mandoline::CutCellMesh<2> &ccm,
                                 bool delaminate) {
    return MandolineVEMMesh2(ccm, .2, delaminate);
}
// VEMMesh3 from_mandoline(const mandoline::CutCellMesh<3>& ccm) { return
// {}; }

MandolineVEMMesh2 from_mandoline(const Eigen::AlignedBox<double, 2> &bb, int nx,
                                 int ny, const mtao::ColVecs2d &V,
                                 const mtao::ColVecs2i &E, bool delaminate) {
    mtao::geometry::grid::StaggeredGrid2d g =
        mtao::geometry::grid::StaggeredGrid2d::from_bbox(
            bb, std::array<int, 2>{{nx, ny}});

    auto ccm = mandoline::construction::from_grid(V, E, g);

    return from_mandoline(ccm, delaminate);
}
std::set<int> MandolineVEMMesh2::boundary_edge_indices() const {
    if (boundary_edges.empty()) {
        return mandoline::operators::region_boundaries(_ccm);
    } else {
        return boundary_edges;
        // return VEMMesh2::boundary_edge_indices();
    }
}
}  // namespace vem
