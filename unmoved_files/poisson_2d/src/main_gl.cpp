#include <igl/parula.h>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

#include <mtao/geometry/mesh/stack_meshes.hpp>
#include <optional>

#include "cutmesh2_to_vemmesh.hpp"
#include "vem/from_simplicial_matrices.hpp"
#include "vem/mesh.hpp"
#include "sim.h"
#include "sim_vis.hpp"

class VemViewer2d : public omtao::opengl::Window2 {
  public:
    int color_mode = 0;
    std::optional<vem::VEMMesh2> vem;
    enum class MeshBCType : int {
        None = 0,
        Dirichlet = 1,
        Neumann = 2,
        LinearDirichlet = 3
    };
    MeshBCType mesh_bc_type = MeshBCType::LinearDirichlet;


    // rendering stuff
    mtao::ColVectors<double, 4> colors;
    mtao::VecXd vertex_values;
    bool vertex_valued_cell_colors = false;
    std::vector<float> histogram;
    size_t nonfinite_entries = 0;
    float val_min, val_max;

    VemViewer2d(const Arguments &args)
      : CurveDrawingViewer2d(args),
        constructor(_flat_shader, background_drawgroup) {
        constructor.setParent(&root());

        sim_vis.setParent(&root());

        cutcell_drawable =
          new mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>{
              cutcell_mesh, vcolor_shader, background_drawgroup
          };
        cutcell_drawable->deactivate();
        cutcell_mesh.setParent(&root());

        // dirichlet_condition_drawable =
        //    new mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>{
        //        dirichlet_condition_mesh, vcolor_shader,
        //        background_drawgroup};
        // dirichlet_condition_drawable->deactivate();
        // dirichlet_condition_mesh.setParent(&root());

        if (args.argc > 1) {
            Corrade::Utility::Arguments myargs;
            myargs.addArgument("filename").parse(args.argc, args.argv);
            std::string filename = myargs.value("filename");
            spdlog::warn("Reading file [{}]", filename);
            curve.load(filename);
            auto p = curve.points();

            update_curve();
            ui_mode = InterfaceMode::Browse;
        }
    }
    void gui() override;
    void update_ccm();
    void update_vem();
    void update_sim();
    void clear_ccm();
    void clear_vem();
    void clear_sim();
    void draw() override;
    void update_curve();
    void solve_vem();
    void mouseMoveEvent(MouseMoveEvent &event) override;

    void update_faces();
    void update_colors();// doesnt upload the colors to anything
  private:
    Magnum::Shaders::Flat2D _flat_shader;
    Magnum::Shaders::VertexColor2D vcolor_shader;
    Magnum::SceneGraph::DrawableGroup2D background_drawgroup;

    mandoline::construction::CutmeshGenerator2Gui constructor;

    // mtao::opengl::objects::Mesh<2> dirichlet_condition_mesh;
    mtao::opengl::objects::Mesh<2> cutcell_mesh;
    mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D> *cutcell_drawable =
      nullptr;
    // mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>
    // *dirichlet_drawable =
    //    nullptr;
};

void VemViewer2d::draw() {
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::FaceCulling);
    Magnum::GL::Renderer::setPointSize(10);

    camera().draw(background_drawgroup);
    camera().draw(sim_vis.drawgroup);
    CurveDrawingViewer2d::draw();
}
void VemViewer2d::mouseMoveEvent(MouseMoveEvent &event) {
    CurveDrawingViewer2d::mouseMoveEvent(event);
    if (event.isAccepted()) {
        return;
    }
    if (ccm) {
        int fi = ccm->cell_index(mtao::Vec2d(cursor.x(), cursor.y()));
        if (fi >= 0) {
            face_index = fi;
        } else {
            face_index.reset();
        }
        // if (ofi != face_index) {
        //    update_face(face_index);
        //}
    }

    // event.setAccepted();
}

void VemViewer2d::gui() {
    CurveDrawingViewer2d::gui();
    if (face_index) {
        ImGui::Text("CCM Face index: %d", *face_index);
    } else {
        ImGui::Text("CCM has no face index");
    }
    if (constructor.gui()) {
        auto V = curve.points().cast<float>().eval();
        auto E = curve.edges();
        if (V.size() > 0) {
            constructor.set_mesh(V.cast<double>(), E.cast<int>());
        }
    }
    { sim_vis.gui(); }
    if (ImGui::Button("Make CCM")) {
        update_ccm();
    }
    if (ccm) {
        if (ImGui::Button("Make VEM")) {
            update_vem();
        }
    } else {
        ImGui::Text("Make CCM for VEM button");
    }
    //    if (!std::holds_alternative<nullptr_t>(vem)) {
    if (vem) {
        if (ImGui::Button("Solve")) {
            solve_vem();
        }
        if (ImGui::Button("Make SIM")) {
            update_sim();
        }
    } else {
        ImGui::Text("Make VEM for solve button");
        ImGui::Text("Make VEM for SIM button");
    }

    if (ccm) {
        //{
        //    const char *items[] = {"DIWLevin", "DIWLevinIntegrated",
        //                           "Pointwise"};
        //    if (ImGui::Combo("VEM Type", reinterpret_cast<int *>(&vem_type),
        //                     items,
        //                     // if (ImGui::Combo("VEM Type", static_cast<int
        //                     // *>(&vem_type), items,
        //                     3)) {
        //        vem.emplace<nullptr_t>();
        //    }
        //}
        {
            const char *items[] = { "None", "Dirichlet", "Neumann", "LinearDirichlet" };
            if (ImGui::Combo("Boundary condition type",
                             reinterpret_cast<int *>(&mesh_bc_type),
                             items,
                             // if (ImGui::Combo("VEM Type", static_cast<int
                             // *>(&vem_type), items,
                             4)) {
                // vem.emplace<nullptr_t>();
            }
        }
    } else {
        ImGui::Text("VEM Type Dialog here");
    }
    if (vem) {
        // std::visit(
        //    [&](auto &&vem) {
        //        using T = std::decay_t<decltype(vem)>;
        //        if constexpr (std::is_same_v<T, DIWLevinVEMMesh2>) {
        //            int order = vem.order;
        //            if (ImGui::InputInt("Max Polynomial Order", &order)) {
        //                vem.order = order;
        //            }

        //        } else if constexpr (std::is_same_v<T, PointwiseVEMMesh2>) {
        int order = vem->order;
        if (ImGui::InputInt("Max Polynomial Order", &order)) {
            vem->order = order;
        }

        int desired_edge_counts = vem->desired_edge_counts;
        if (ImGui::InputInt("Edge subsamples", &desired_edge_counts)) {
            vem->desired_edge_counts = desired_edge_counts;
            vem->initialize_interior_offsets();
        }
        float lambda = vem->projection_lambda;
        if (ImGui::InputFloat("Projection lambda", &lambda, 1, 10000)) {
            vem->projection_lambda = lambda;
        }
        //       } else {
        //           ImGui::Text("VEM Options here");
        //       }
        //   },
        //   vem);
    }
    if (histogram.size() > 0) {
        ImGui::Text("Range %f => %f", val_min, val_max);
        ImGui::PlotHistogram("value histogram", histogram.data(), histogram.size(), 0, nullptr, FLT_MIN, FLT_MAX, ImVec2(0, 80.f));
        if (nonfinite_entries > 0) {
            ImGui::Text("Nonfinite entry count: %d", int(nonfinite_entries));
        }
    }
}

void VemViewer2d::update_curve() {
    CurveDrawingViewer2d::update_curve();
    auto V = curve.points().cast<float>().eval();
    auto E = curve.edges();
    if (V.size() > 0) {
        constructor.set_mesh(V.cast<double>(), E.cast<int>());
    }
}
void VemViewer2d::update_ccm() {
    face_index.reset();
    vertex_valued_cell_colors = false;
    color_mode = 0;
    constructor.bake();
    ccm = constructor.emit();

    {
        auto V = ccm->vertices();
        auto E = ccm->cut_edges_eigen();

        cutcell_mesh.setEdgeBuffer(V.cast<float>().eval(),
                                   E.cast<unsigned int>().eval());
        cutcell_drawable->deactivate();

        spdlog::warn("Making gird draable");
        // ccm->faces();
        auto r = ccm->regions();
        spdlog::warn("Writing regions");
        std::set<int> regions;
        for (int i = 0; i < r.size(); ++i) {
            regions.insert(r(i));
        }
        std::cout << "Region count: " << regions.size() << std::endl;
    }
    update_faces();
}
void VemViewer2d::update_vem() {
    if (!ccm) {
        spdlog::warn("Cannot make a VEM mesh without a CCM mesh first!");
        return;
    }

    // switch (vem_type) {
    //    case VEMType::DIWLevin:
    //    case VEMType::DIWLevinIntegrated:
    //        vem.emplace<DIWLevinVEMMesh2>();
    //        break;
    //    case VEMType::Pointwise:
    //        vem.emplace<PointwiseVEMMesh2>();
    //}
    vem = std::make_unique<PointwiseVEMMesh2>();
    // std::visit(
    //    [&](auto &&vem) {
    //        using T = std::decay_t<decltype(vem)>;
    //        if constexpr (std::is_same_v<nullptr_t, T>) {
    //            return;
    //        } else {
    //            if constexpr (std::is_same_v<T, DIWLevinVEMMesh2>) {
    //                vem.integrated_edges =
    //                    vem_type == VEMType::DIWLevinIntegrated;
    //            }
    vem->order = 3;
    cutmesh2_to_vemmesh(*ccm, *vem);
    vem->initialize_interior_offsets(0);
    //            // std::cout << "vem edges:\n" << vem.edges << std::endl;
    //            // for(auto&& [cidx, cell]:
    //            // mtao::iterator::enumerate(vem.cells)) { std::cout <<
    //            // "CEll: "
    //            // << cidx << std::endl; for(auto&& [a,b]: cell) { std::cout
    //            // << a << ":" << b << " ";
    //            //}
    //            // std::cout << std::endl;
    //            //}
    //        }
    //    },
    //    vem);
    color_mode = 1;

    vertex_valued_cell_colors = false;
    update_faces();
}
void VemViewer2d::solve_vem() {
    if (!vem || !ccm) {
        spdlog::warn(
          "Cannot make a solution without a VEM mesh and CCM mesh "
          "first!");
        return;
    }

    // make RHS

    std::map<size_t, double> dirichlet_vertices;
    std::map<size_t, double> neumann_edges;

    auto grid = ccm->vertex_grid();
    auto grid_shape = grid.shape();
    int max = grid_shape[1] - 1;
    // set dirichlet boundary conditions on teh boundary
    if (mesh_bc_type == MeshBCType::Dirichlet || mesh_bc_type == MeshBCType::LinearDirichlet) {
        for (size_t j = 0; j < grid_shape[0]; ++j) {
            dirichlet_vertices[grid.index(j, 0)] = 1;
            dirichlet_vertices[grid.index(j, max)] = -1;
        }
    }
    if (mesh_bc_type == MeshBCType::Dirichlet) {
        // set dirichlet boundary conditions on the mesh
        std::set<int> vertices;
        for (auto &&f : ccm->cut_faces()) {
            if (f.is_mesh_face()) {
                for (auto &&loop : f.indices) {
                    std::copy(loop.begin(), loop.end(), std::inserter(vertices, vertices.end()));
                }
            }
        }
        for (auto &&v : vertices) {
            dirichlet_vertices[v] = 0;
        }
    }
    if (mesh_bc_type == MeshBCType::Neumann) {
        for (auto &&[ceidx, ce] : mtao::iterator::enumerate(ccm->cut_edges())) {
            if (ce.is_mesh_edge()) {
                auto a = ccm->vertex(ce.indices[0]);
                auto b = ccm->vertex(ce.indices[1]);
                auto N = ce.N;
                neumann_edges[ceidx] = N(0) * (b - a).norm();
            }
        }
    }
    // std::visit(
    //    [&](auto &&vem) {
    //        using T = std::decay_t<decltype(vem)>;
    //        if constexpr (!std::is_same_v<nullptr_t, T>) {
    vertex_values = vem->laplace_problem(dirichlet_vertices, neumann_edges);
    //        }
    //    },
    //    vem);
    vertex_values = vertex_values.array().isFinite().select(vertex_values, 0);

    histogram.resize(50);
    {
        val_min = vertex_values.minCoeff();
        val_max = vertex_values.maxCoeff();
        std::fill(histogram.begin(), histogram.end(), 0);
        nonfinite_entries = 0;

        for (int i = 0; i < vertex_values.size(); ++i) {
            double v = vertex_values(i);
            if (std::isfinite(v)) {
                int pos = 50 * (v - val_min) / (val_max - val_min);
                pos = std::clamp<int>(pos, 0, histogram.size() - 1);
                histogram[pos]++;
            } else {
                spdlog::warn("Nonfinite value");
                nonfinite_entries++;
            }
        }
    }

    vertex_valued_cell_colors = true;

    color_mode = 2;
    update_faces();
}
void VemViewer2d::update_faces() {
    if (!ccm) return;

    spdlog::warn("updating faces");

    update_colors();
    spdlog::warn("making meshes");
    auto verts = ccm->vertices();
    if (vertex_valued_cell_colors) {
        std::vector<mtao::ColVecs3i> Fs;
        Fs.reserve(ccm->num_cells());
        for (int idx = 0; idx < ccm->num_cells(); ++idx) {
            auto c = ccm->cell(idx);
            if (c.size() > 0) {
                auto &&d = *c.begin();
                auto F = mtao::geometry::mesh::earclipping(verts, d);
                Fs.emplace_back(std::move(F));
            }
            // spdlog::info("{} / {}", idx, ccm->num_cells());
        }
        //});

        auto F = mtao::eigen::hstack_iter(Fs.begin(), Fs.end());
        spdlog::warn("stacking meshes");
        spdlog::error("nV/nF/nC: {} {} {}", verts.cols(), F.cols(), colors.cols());

        cutcell_mesh.setTriangleBuffer(verts.cast<float>(),
                                       F.cast<unsigned int>());
        cutcell_mesh.setColorBuffer(colors.cast<float>().eval());
        cutcell_drawable->activate_triangles();
    } else {
        std::vector<std::tuple<mtao::ColVecs3i, mtao::Vec4d>> FCs;
        FCs.resize(ccm->num_cells());
        // tbb::parallel_for(tbb::blocked_range<size_t>(0,
        // ccm->num_cells()),
        // [&](const tbb::blocked_range<size_t> &range) {
        //    for (size_t idx = range.begin(); idx != range.end(); ++idx) {
        for (int idx = 0; idx < ccm->num_cells(); ++idx) {
            // spdlog::info("Getting cell {}", idx);
            auto c = ccm->cell(idx);
            if (c.size() > 0) {
                auto &&d = *c.begin();
                // std::copy(d.begin(),d.end(),std::ostream_iterator<int>(std::cout,","));
                // std::cout << std::endl;

                // std::cout << idx << " => " <<
                // mtao::geometry::curve_volume(verts, d) << std::endl;
                auto F = mtao::geometry::mesh::earclipping(verts, d);
                FCs[idx] = std::tuple<mtao::ColVecs3i, mtao::Vec4d>{
                    std::move(F), colors.col(idx)
                };
            }
            // spdlog::info("{} / {}", idx, ccm->num_cells());
        }
        //});

        spdlog::warn("stacking meshes");
        auto [V, F, C] =
          mtao::geometry::mesh::stack_meshes(ccm->vertices(), FCs);
        spdlog::error("nV/nF/nC: {} {} {}", V.cols(), F.cols(), C.cols());

        cutcell_mesh.setTriangleBuffer(V.cast<float>(), F.cast<unsigned int>());
        cutcell_mesh.setColorBuffer(C.cast<float>().eval());
        cutcell_drawable->activate_triangles();
    }
}
void VemViewer2d::update_colors() {
    spdlog::warn("updating colors");
    if (!ccm) {
        spdlog::warn("No ccm, not actually updating colors");
        return;
    }
    colors.resize(4, ccm->num_cells());
    auto region_colors = [&]() {
        spdlog::warn("computing region colors");
        mtao::VecXi R = ccm->regions();
        int num_regions = R.maxCoeff() + 1;
        mtao::ColVecs4d C(4, num_regions);
        C.setRandom();
        C.row(3).setConstant(1);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, R.size()),
                          [&](const tbb::blocked_range<size_t> &range) {
                              for (size_t idx = range.begin();
                                   idx != range.end();
                                   ++idx) {
                                  colors.col(idx) = C.col(R(idx));
                              }
                          });
    };
    auto random_colors = [&]() {
        spdlog::warn("computing random colors");
        colors.setRandom();
    };

    auto vertex_colors = [&]() {
        Eigen::MatrixXd cols;
        igl::parula(vertex_values, -1, 1, cols);
        igl::parula(vertex_values, true, cols);
        colors.resize(4, vertex_values.size());
        colors.topRows<3>() = cols.transpose();
        colors.row(3).setConstant(1);
    };

    switch (color_mode) {
    case 1:
        region_colors();
        break;
    case 0:
    default:
        random_colors();
        break;
    case 2:
        vertex_colors();
    }
    spdlog::warn("Picked out colors");
    colors.noalias() = (colors.array() > 0).select(colors, 0);
    colors.row(3).setConstant(1);
}
void VemViewer2d::update_sim() {
    if (!ccm || !vem) {
        spdlog::warn("Cannot create a sim without a CCM and a VEM ({} {})",
                     bool(ccm),
                     bool(vem));
        // bool(ccm), !std::holds_alternative<nullptr_t>(vem));
        return;
    }
    // std::visit(
    //    [&](auto &&vem) {
    //        using T = std::decay_t<decltype(vem)>;
    //        if constexpr (!std::is_same_v<T, nullptr_t>) {
    if (vem) {
        sim = std::make_unique<Sim>(*ccm, *vem);
    }
    //},
    // vem);
    if (sim) {
        sim->initialize_particles(1000, [](const mtao::Vec2d &p) {
            return mtao::Vec2d(std::cos(5 * p.x()), std::cos(5 * p.y()));
        });
        sim->update_particle_cell_cache();
        sim->particle_velocities_to_field();
        sim_vis.set_sim(sim);
        sim_vis.update();
    }
}

void VemViewer2d::clear_ccm() {
    clear_vem();
    ccm.reset();
}
void VemViewer2d::clear_vem() {
    clear_sim();
    // vem.emplace<nullptr_t>();
    vem.reset();
}
void VemViewer2d::clear_sim() {
    sim_vis.clear();
    sim.reset();
}

MAGNUM_APPLICATION_MAIN(VemViewer2d)
