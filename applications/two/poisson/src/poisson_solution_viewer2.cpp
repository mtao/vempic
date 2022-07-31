#include <Magnum/EigenIntegration/Integration.h>
#include <Magnum/GL/Renderer.h>
#include <mtao/opengl/Window.h>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <mtao/geometry/mesh/stack_meshes.hpp>
#include <mtao/opengl/shaders/polynomial_scalar_field.hpp>
#include <mtao/opengl/shaders/vector_field.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <optional>
#include <vem/two/monomial_field_embedder.hpp>
#include <vem/two/poisson/constraint_viewer.hpp>
#include <vem/two/poisson/example_constraints.hpp>
#include <vem/two/poisson/poisson.hpp>
#include <vem/two/boundary_facets.hpp>
#include <vem/two/visualize/vem_mesh_creation_gui.hpp>
#include <vem/two/visualize/vem_scalar_field_viewer.hpp>

class VemViewer2d : public mtao::opengl::Window2 {
   public:
    std::shared_ptr<const vem::two::VEMMesh2> mesh;
    std::optional<vem::two::poisson::PoissonVEM2> poisson_vem;
    std::optional<vem::two::MonomialVectorFieldEmbedder> vector_field;
    std::optional<vem::two::poisson::ScalarConstraintsGui> constraints;
    // set of cells used for each region
    std::vector<std::set<int>> cell_regions;
    // our currently chosen region
    std::optional<int> active_cell_region_index;
    int system_degree = 1;
    float timestep = 0.02;

    VemViewer2d(const Arguments &args);
    mtao::ColVecs2d points;
    mtao::VecXi point_cells;
    bool show_mesh_selection_window = false;

    void gui() override;
    void draw() override;
    void mouseMoveEvent(MouseMoveEvent &event) override;
    void set_pointwise_function(const std::function<double(double)> &f);

   private:
    mtao::opengl::PolynomialScalarFieldShader<2> poly_shader;
    std::optional<vem::two::visualize::VEM2ScalarFieldViewer> pmesh;
    Magnum::SceneGraph::DrawableGroup2D post_mesh_drawables;

    vem::two::visualize::VEMMesh2CreationGui mesh_gui;
    mtao::opengl::objects::Mesh<2> boundary_mesh;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *boundary_mesh_drawable = nullptr;

    Magnum::Shaders::Flat2D _flat_shader;
    mtao::opengl::objects::Mesh<2> cursor_mesh;
#ifdef USE_VFIELD

    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>
        *cursor_drawable =
#else
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D> *cursor_drawable =
#endif
            nullptr;

    mtao::opengl::VectorFieldShader<2> vf_shader;
    mtao::opengl::objects::Mesh<2> vfield_mesh;
    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>> *vf_viewer =
        nullptr;

    float poly_constant = 0;
    mtao::Vec2f poly_linear = mtao::Vec2f(1.f, 0.f);

    int pulled_index = 0;
    float pulled_value = 1.0;

    mtao::Vec2f cursor;
    const std::set<int> &active_cell_regions() const;

    void refresh_mesh_visualization();
    void cell_regions_updated();
    void solve_constraints();
    void update_constraint_view();
    void view_boundary_vertices();

   public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
VemViewer2d::VemViewer2d(const Arguments &args)
    : Window2(args), mesh_gui(&scene(), &post_mesh_drawables) {
    boundary_mesh.setParent(&scene());
    // boundary_drawable = new
    // mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>{
    //    boundary_mesh, _flat_shader, post_mesh_drawables};
    boundary_mesh_drawable = new mtao::opengl::MeshDrawable{
        boundary_mesh, _flat_shader, post_mesh_drawables};
    boundary_mesh_drawable->data().color = Magnum::Color4(0, 0, 0, 1);
    boundary_mesh_drawable->deactivate();
    boundary_mesh_drawable->line_width = 5;
    // std::string a("../alligator_fixed.obj ");
    // std::copy(a.begin(), a.end(), mesh_filename);
    //// mesh_filename[a.size()+1] = '\0';
    // make_mandoline_mesh();

    cursor_mesh.setParent(&scene());
#ifdef USE_VFIELD
    cursor_drawable = new mtao::opengl::MeshDrawable{cursor_mesh, vf_shader,
                                                     post_mesh_drawables};
#else
    cursor_drawable = new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>{
        cursor_mesh, _flat_shader, post_mesh_drawables};
#endif
    cursor_mesh.setVertexBuffer(mtao::Vec2f::Zero().eval());
    cursor_mesh.setVFieldBuffer(mtao::Vec2f::Zero().eval());
    cursor_drawable->data().color = Magnum::Color4(1, 1, 1, 1);
    cursor_drawable->deactivate();
    cursor_drawable->activate_points();
    cursor_drawable->point_size = 5;
}

void VemViewer2d::refresh_mesh_visualization() {
    if (!(bool(mesh))) {
        spdlog::error("Cannot refresh mesh visualization without a mesh");
    }
    cell_regions = mesh->cell_regions();
    poisson_vem.emplace(*mesh, system_degree);
    pmesh.emplace(*mesh,
                  std::make_shared<vem::two::MonomialBasisIndexer>(
                      poisson_vem->monomial_indexer()),
                  &drawables());
    vector_field.emplace(poisson_vem->monomial_indexer());
    {
        auto P = poisson_vem->polynomial_to_sample_evaluation_matrix();
        auto M = poisson_vem->sample_to_polynomial_projection_matrix();
    }

    mtao::VecXd D = (mesh->V.colwise() - mesh->V.rowwise().mean().eval())
                        .colwise()
                        .squaredNorm()
                        .transpose();

    D.conservativeResize(poisson_vem->system_size());
    // D = mesh->V.row(1).transpose();
    // D = mesh->V.array().colwise().sum().transpose();
    pmesh->setPointValues(D);

    mtao::VecXd P = poisson_vem->sample_to_polynomial_projection_matrix() * D;

    // std::cout << P << std::endl;
    std::cout << P.rows() << std::endl;
    pmesh->set_coefficients(P);
    pmesh->set_scales(mtao::eigen::stl2eigen(
        poisson_vem->monomial_indexer().diameters()));
    pmesh->set_degrees(
        mtao::eigen::stl2eigen(poisson_vem->monomial_indexer().degrees())
            .cast<int>());

    pmesh->set_degrees(mtao::VecXi::Constant(mesh->cell_count(), 2));
    pmesh->setParent(&root());

    constraints.emplace(*pmesh, &post_mesh_drawables);
    constraints->setParent(&root());
    vfield_mesh.setParent(&root());
    if (vf_viewer == nullptr) {
        vf_viewer =
            new mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>{
                vfield_mesh, vf_shader, post_mesh_drawables};
    }
    vf_viewer->deactivate();
    vfield_mesh.setVertexBuffer(mesh->V.cast<float>().eval());

    {
        boundary_mesh.setVertexBuffer(mesh->V.cast<float>().eval());
        auto edge_indices = mesh->boundary_edge_indices();
        if (edge_indices.size() > 0) {
            mtao::ColVectors<unsigned int, 2> E(2, edge_indices.size());
            E.setZero();
            const auto &CE = mesh->E;
            for (auto &&[a, b] : mtao::iterator::enumerate(edge_indices)) {
                E.col(a) = CE.col(b).cast<unsigned int>();
            }

            boundary_mesh.setEdgeBuffer(E);
            boundary_mesh_drawable->activate_edges();
        } else {
            boundary_mesh_drawable->deactivate();
        }
    }
}

void VemViewer2d::draw() {
    // Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);
    // Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::FaceCulling);
    // Magnum::GL::Renderer::setPointSize(10);

    // camera().draw(background_drawgroup);
    // camera().draw(sim_vis.drawgroup);
    mtao::opengl::Window2::draw();
    camera().draw(post_mesh_drawables);
}
void VemViewer2d::mouseMoveEvent(MouseMoveEvent &event) {
    mtao::opengl::Window2::mouseMoveEvent(event);
    auto c = localPosition(event.position());
    cursor = Magnum::EigenIntegration::cast<mtao::Vec2f>(c);

    cursor_mesh.setVertexBuffer(cursor);
    if (vector_field) {
        mtao::ColVecs2d P(2, 100);
        P.col(0) = cursor.cast<double>();
        for (int j = 1; j < P.cols(); ++j) {
            auto po = P.col(j - 1);
            auto p = P.col(j);
            p = vector_field->advect_rk2(po, timestep);
        }
        cursor_mesh.setVertexBuffer(P.cast<float>().eval());
        auto v = vector_field->get_vector(cursor.cast<double>())
                     .cast<float>()
                     .eval();
        cursor_mesh.setVFieldBuffer(v);
    }
    if (event.isAccepted()) {
        return;
    }

    // event.setAccepted();
}

void VemViewer2d::gui() {
    ImGui::Text("Cursor Position: (%f,%f)", cursor.x(), cursor.y());
    if (ImGui::InputInt("VEM poly degree", &system_degree)) {
        refresh_mesh_visualization();
    }
    ImGui::InputFloat("Timestep", &timestep);
    if (pmesh) {
        pmesh->gui();
    }
    if (vf_viewer) {
        vf_viewer->gui();
    }
    if (cursor_drawable) {
        cursor_drawable->gui("Cursor Viz Params");
    }

    if (mesh) {
        {
            if (active_cell_region_index) {
                if (ImGui::InputInt("Active region id",
                                    &*active_cell_region_index)) {
                    active_cell_region_index = std::clamp<int>(
                        *active_cell_region_index, 0, cell_regions.size() - 1);
                    cell_regions_updated();
                }
                if (ImGui::Button("Use all regions")) {
                    active_cell_region_index = {};
                    cell_regions_updated();
                }
            } else {
                if (ImGui::Button("Select a cell region")) {
                    active_cell_region_index = 0;
                    cell_regions_updated();
                }
            }
        }

        if (ImGui::Button("View boundary vertices")) {
            view_boundary_vertices();
        }
        if (ImGui::Button("Clear Constraints")) {
            constraints->clear();
        }
        bool changed = false;

        if (ImGui::InputFloat("Constraint Polynomial Constant",
                              &poly_constant)) {
        }
        if (ImGui::InputFloat2("Constraint Polynomial Linear",
                               poly_linear.data())) {
        }

        if (ImGui::Button("Linear Dirichlet Boundaries")) {
            *constraints = vem::two::poisson::linear_function_dirichlet(
                *mesh, poly_constant, poly_linear.cast<double>());
            update_constraint_view();
        }
        if (ImGui::Button("Linear Neumann")) {
            *constraints = vem::two::poisson::linear_function_neumann(
                *mesh, poly_constant, poly_linear.cast<double>());
            update_constraint_view();
        }
        if (ImGui::Button("Zero Dirichlet Boundary")) {
            *constraints = vem::two::poisson::linear_function_dirichlet(
                *mesh, 0, mtao::Vec2d::Zero());
            update_constraint_view();
        }
        if (constraints->gui(*mesh)) {
            update_constraint_view();
        }
        if (ImGui::Button("Solve Constraints")) {
            solve_constraints();
        }
    }
    ImGui::Checkbox("Show Mesh Selection Window", &show_mesh_selection_window);
    if (show_mesh_selection_window) {
        if (mesh_gui.gui()) {
            mesh = mesh_gui.stored_mesh();
            refresh_mesh_visualization();
        }
    }
}

void VemViewer2d::update_constraint_view() {
    if (!constraints) {
        return;
    }
    mtao::VecXd dirichlet_values(mesh->vertex_count());
    dirichlet_values.setZero();
    for (auto &&[p, v] : constraints->pointwise_dirichlet) {
        dirichlet_values(p) = v;
    }
    pmesh->setPointValues(dirichlet_values);
}

void VemViewer2d::solve_constraints() {
    if (!bool(mesh)) {
        spdlog::error("Cannot solve constraints without constraints");
    }
    if (!bool(poisson_vem)) {
        spdlog::error("Cannog solve constraints without poisson_vem");
    }
    spdlog::info("Generating stiffness matrix and kkt system");
    auto [A, b] = poisson_vem->kkt_system(*constraints, active_cell_regions());
    spdlog::info("Finished generating stiffness matrix and kkt system");
    // std::cout << "KKT matrix\n";
    // std::cout << A << std::endl;

    mtao::VecXd x = b;
    // std::cout << "Right hand side: " << std::endl;
    // std::cout << b.transpose() << std::endl;

    x.setZero();

    spdlog::info("Solving kkt system");
    mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-10);
    spdlog::info("Finished solving kkt system");
    // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
    // x = solver.solve(b);

    // std::cout << "Raw x: " << x.transpose() << std::endl;

    x = x.head(poisson_vem->system_size()).eval();
    // std::cout << "Lambda-free x: " << x.transpose() << std::endl;
    // std::cout << "Residual: " << (A * x - b).transpose() << std::endl;
    // std::cout << "Error norm: " << (A * x - b).norm() << std::endl;
    mtao::VecXd P = poisson_vem->sample_to_polynomial_projection_matrix(
                        active_cell_regions()) *
                    x;

    // std::cout << "Coefficients: " << P.transpose() << std::endl;
    pmesh->setPointValues(x.head(mesh->vertex_count()));
    pmesh->set_coefficients(P);

    {
        auto S2G = poisson_vem->sample_to_poly_gradient(active_cell_regions());
        mtao::VecXd GV = S2G * x;
        int size = poisson_vem->monomial_indexer().num_coefficients();
        mtao::ColVecs2d G(2, size);
        G.row(0) = GV.head(size).head(G.cols()).transpose();
        G.row(1) = GV.tail(size).head(G.cols()).transpose();
        // std::cout << G << std::endl;
        vector_field->set_coefficients(G);
    }

    {
        auto S2SG =
            poisson_vem->sample_to_sample_gradient(active_cell_regions());
        mtao::VecXd GV = S2SG * x;
        mtao::ColVecs2f G(2, mesh->vertex_count());
        int sys_size = poisson_vem->system_size();
        G.row(0) = GV.head(sys_size).head(G.cols()).transpose().cast<float>();
        G.row(1) = GV.tail(sys_size).head(G.cols()).transpose().cast<float>();
        // std::cout << G << std::endl;
        vfield_mesh.setVFieldBuffer(G);

        vf_viewer->activate_points();
    }
}

void VemViewer2d::view_boundary_vertices() {
    auto boundary_vertices = vem::two::boundary_vertices(*mesh);
    auto boundary_edges_map = boundary_edge_map(*mesh);
    std::set<size_t> boundary_edges;
    for (auto &&[a, b] : boundary_edges_map) {
        boundary_edges.emplace(a);
    }
    spdlog::info("{} boundary edges, {} boundary vertices",
                 boundary_edges.size(), boundary_vertices.size());
    spdlog::info("Edges {}", fmt::join(boundary_edges, ","));
    for (auto &&e : boundary_edges) {
        std::cout << mesh->E.col(e).transpose() << std::endl;
    }

    spdlog::info("Vertices {}", fmt::join(boundary_vertices, ","));
    mtao::VecXd D(mesh->V.cols());
    D.setZero();
    for (auto &&v : boundary_vertices) {
        D(v) = 1;
    }
    pmesh->setPointValues(D);
}
void VemViewer2d::cell_regions_updated() {
    if (pmesh) {
        pmesh->active_cells = active_cell_regions();
    }
}
const std::set<int> &VemViewer2d::active_cell_regions() const {
    const static std::set<int> empty = {};

    if (active_cell_region_index) {
        int index = *active_cell_region_index;
        if (index >= 0 || index < cell_regions.size()) {
            return cell_regions.at(index);
        }
    }

    return empty;
}
MAGNUM_APPLICATION_MAIN(VemViewer2d)
