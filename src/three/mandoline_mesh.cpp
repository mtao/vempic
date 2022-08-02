
#include "vem/three/mandoline_mesh.hpp"

#include <mandoline/construction/construct.hpp>
#include <mandoline/operators/volume3.hpp>
#include <mtao/logging/stopwatch.hpp>

//#include "vem/utils/merge_cells.hpp"

namespace vem::three {

double MandolineVEMMesh3::dx() const {
    auto dx = _ccm.dx();
    return *std::min_element(dx.begin(), dx.end());
}
int MandolineVEMMesh3::grade(size_t cell_index) const {
    return collision_free(cell_index) ? 0 : 1;
}
std::string MandolineVEMMesh3::type_string() const { return "cutmesh"; }

// MandolineVEMMesh3::MandolineVEMMesh3(mandoline::CutCellMesh<3> ccm)
MandolineVEMMesh3::MandolineVEMMesh3(const mandoline::CutCellMesh<3> &ccm)
    : _ccm(std::move(ccm)), _ccm_cell_parents(_ccm) {
    _ccm.cache_vertices();
    this->V = ccm.vertices();
    spdlog::info("Should be getting cell centroids");
    this->C = ccm.cell_centroids();
    this->FC = ccm.face_centroids();
    this->face_frames.resize(ccm.num_faces());

    for (auto &&[face_index_, frame_] :
         mtao::iterator::enumerate(face_frames)) {
        const auto &face_index = face_index_;
        auto &frame = frame_;
        auto set_from_grid_axis = [&](int axis) {
            frame.col(0).setUnit((axis + 1) % 3);
            frame.col(1).setUnit((axis + 2) % 3);
        };
        if (_ccm.is_cut_face(face_index)) {
            auto &&f = _ccm.cut_faces().at(face_index);
            if (f.is_mesh_face()) {
                int mfid = f.as_face_id();
                auto mf = ccm.origF().col(mfid);
                auto a = ccm.origV().col(mf(0));
                auto b = ccm.origV().col(mf(1));
                auto c = ccm.origV().col(mf(2));

                auto x = frame.col(0) = (b - a).normalized();
                auto y = frame.col(1) = c - a;
                y -= x.dot(y) * x;
                y.normalize();
            } else {
                int axis = f.as_axial_axis();
                set_from_grid_axis(axis);
            }
        } else {
            const mandoline::AdaptiveGrid::Face &f =
                _ccm.exterior_grid().face(face_index - _ccm.num_cut_faces());

            set_from_grid_axis(f.axis());
        }
    }
    _ccm.triangulate_faces(false);

    auto B3 = _ccm.boundary(true);

    this->cell_boundary_map.resize(_ccm.num_cells());
    spdlog::info(
        "CCM had {} cells ({} cut-cells and {} exterior cells), i have {}",
        _ccm.num_cells(), _ccm.num_cut_cells(),
        _ccm.exterior_grid().num_cells(), this->cell_boundary_map.size());

    for (int o = 0; o < B3.outerSize(); ++o) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B3, o); it; ++it) {
            if (!_ccm.is_folded_face(it.row())) {
                bool value = it.value() < 0;
                this->cell_boundary_map[it.col()][it.row()] = value;
            }
        }
    }
    if (true) {
        std::set<int> BEI;
        for (auto &&[fidx, f] : ccm.mesh_cut_faces()) {
            BEI.emplace(fidx);
        }

        std::set<int> seen_faces;
        std::set<int> to_delaminate;

        // first pass to find which faces do need to be delaminated
        for (auto &&c : this->cell_boundary_map) {
            for (auto &&[eidx, sgn] : c) {
                if (BEI.contains(eidx)) {
                    auto [it, added] = seen_faces.emplace(eidx);
                    if (!added) {
                        to_delaminate.emplace(eidx);
                    }
                }
            }
        }

        face_to_ccm_face_map.resize(_ccm.num_faces() + to_delaminate.size());
        std::iota(face_to_ccm_face_map.begin(), face_to_ccm_face_map.end(), 0);
        int cur_pos = _ccm.num_faces();
        FC.conservativeResize(3, face_count());
        face_frames.resize(face_count());
        // no triangulated_faces yet!
        // E.conservativeResize(2, E.cols() + to_delaminate.size());

        for (auto &&c : this->cell_boundary_map) {
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
                        face_to_ccm_face_map[index] = eidx;

                        face_frames[index] = face_frames[eidx];
                        FC.col(index) = FC.col(eidx);
                        new_c.erase(eidx);
                        new_c[index] = sgn;
                    }
                }
                c = std::move(new_c);
            }
        }
    } else {
        face_to_ccm_face_map.resize(ccm.face_size());
        std::iota(face_to_ccm_face_map.begin(), face_to_ccm_face_map.end(), 0);
    }

    triangulated_faces.resize(face_count());
    for (auto &&[idx, tf] :
         mtao::iterator::enumerate(this->triangulated_faces)) {
        tf = triangulated_face(idx);
    }

    /*
    std::optional<double> merge_cell_volume = .1;
    if (merge_cell_volume) {
        double frac = std::clamp(*merge_cell_volume, 0., 1.);
        double threshold = _ccm.dx().prod() * frac;
        auto vol = mandoline::operators::cell_volumes(_ccm);

        auto [new_cells, old_to_new] = vem::utils::merge_cells(
            cell_boundary_map, vol, mtao::eigen::stl2eigen(_ccm.regions()),
            threshold);

        _ccm_cell_to_cell = std::move(old_to_new);

        mtao::ColVecs3d CNew(3, new_cells.size());
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

        spdlog::info("Merged {} cells to {} cells", cell_boundary_map.size(),
                     new_cells.size());
        cell_boundary_map = std::move(new_cells);
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
        _cell_to_ccm_cells.resize(cell_boundary_map.size());
        _ccm_cell_to_cell.resize(cell_boundary_map.size());
        for (auto &&[i, a] : mtao::iterator::enumerate(_ccm_cell_to_cell)) {
            _cell_to_ccm_cells[a].emplace(i);
        }
    }
    */
}
int MandolineVEMMesh3::face_count() const {
    return face_to_ccm_face_map.size();
}  //_ccm.num_faces(); }
bool MandolineVEMMesh3::in_cell(const mtao::Vec3d &p, int cell_index) const {
    return _ccm.is_in_cell(p, cell_index);
}
int MandolineVEMMesh3::get_cell(const mtao::Vec3d &p, int last_known) const {
    if (_ccm.bbox().contains(p)) {
        return mandoline::operators::nearest_cells(_ccm, _ccm_cell_parents,
                                                   p)(0);
    } else {
        return -1;
    }
    // return _ccm.get_nearest_cell_index(p);
}
mtao::VecXi MandolineVEMMesh3::get_cells(const mtao::ColVecs3d &P,
                                         const mtao::VecXi &last_known) const {
    auto R = mandoline::operators::nearest_cells(_ccm, _ccm_cell_parents, P);
    auto bb = _ccm.bbox();
    for (int j = 0; j < R.size(); ++j) {
        if (!bb.contains(P.col(j))) {
            R(j) = -1;
        }
    }
    return R;
}

bool MandolineVEMMesh3::collision_free(size_t cell_index) const {
    if (_ccm.is_cut_cell(cell_index)) {
        return false;
    }
    auto c = _ccm.StaggeredGrid::cell_grid().unindex(cell_index);

    auto &&s = _ccm.StaggeredGrid::cell_grid().shape();
    for (auto &&[c, s] : mtao::iterator::zip(c, s)) {
        if (c == 0 || c == s - 1) {
            return false;
        }
    }
    return true;
}
PolygonBoundaryIndices MandolineVEMMesh3::face_loops(size_t face_index) const {
    face_index = face_to_ccm_face_map[face_index];
    if (_ccm.is_cut_face(face_index)) {
        auto &&f = _ccm.cut_faces().at(face_index);
        if (f.indices.size() == 1) {
            return *f.indices.begin();
        } else {
            std::set<std::vector<int>> p(f.indices.begin(), f.indices.end());
            auto it = p.begin();
            std::vector<int> p2 = std::move(*it);
            p.erase(it);
            return {p2, p};
        }
    } else {
        const mandoline::AdaptiveGrid::Face &f =
            _ccm.exterior_grid().face(face_index - _ccm.num_cut_faces());

        int a = _ccm.vertex_index(f.vertex(0, 0));
        int b = _ccm.vertex_index(f.vertex(1, 0));
        int c = _ccm.vertex_index(f.vertex(1, 1));
        int d = _ccm.vertex_index(f.vertex(0, 1));
        return {{a, b, c, d}};
    }
}
mtao::Vec3d MandolineVEMMesh3::normal(int face_index) const {
    face_index = face_to_ccm_face_map[face_index];
    if (_ccm.is_cut_face(face_index)) {
        const auto &uv = face_frames.at(face_index);
        return uv.col(0).cross(uv.col(1)).normalized();
        return _ccm.cut_face(face_index).N;
    } else {
        return -mtao::Vec3d::Unit(_ccm.exterior_grid()
                                      .face(face_index - _ccm.cut_face_size())
                                      .axis());
    }
}
std::vector<std::set<int>> MandolineVEMMesh3::cell_regions() const {
    auto R = _ccm.regions();

    int region_count = *std::max_element(R.begin(), R.end()) + 1;
    std::vector<std::set<int>> ret(region_count);
    for (int j = 0; j < R.size(); ++j) {
        ret[R[j]].emplace(j);
    }
    return ret;
}
mtao::ColVecs3i MandolineVEMMesh3::triangulated_face(size_t face_index) const {
    face_index = face_to_ccm_face_map[face_index];
    if (_ccm.is_cut_face(face_index)) {
        auto &&f = _ccm.cut_faces().at(face_index);
        if (!f.triangulation) {
            spdlog::error(
                "FATAL! we expect you to triangulate the CCM before "
                "loading it "
                "into VEM!");
        }
        return *f.triangulation;
    } else {
        const mandoline::AdaptiveGrid::Face &f =
            _ccm.exterior_grid().face(face_index - _ccm.num_cut_faces());

        int a = _ccm.vertex_index(f.vertex(0, 0));
        int b = _ccm.vertex_index(f.vertex(1, 0));
        int c = _ccm.vertex_index(f.vertex(1, 1));
        int d = _ccm.vertex_index(f.vertex(0, 1));
        mtao::ColVecs3i F(3, 2);
        F.col(0) << a, c, b;
        F.col(1) << a, d, c;
        return F;
    }
}


std::optional<int> MandolineVEMMesh3::cell_category(size_t cell_index) const {
    if (_ccm.is_cut_cell(cell_index)) {
        return {};
    } else {
        const auto &cell = _ccm.exterior_grid().cell(cell_index);
        return cell.width();
    }
}
}  // namespace vem
