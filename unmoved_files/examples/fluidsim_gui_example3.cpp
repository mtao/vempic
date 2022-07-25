#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Arguments.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Shaders/VertexColor.h>
#include <mtao/geometry/grid/grid.h>
#include <mtao/opengl/objects/grid.h>
#include <mtao/types.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>
#include <memory>
#include <mtao/opengl/shaders/vector_field.hpp>

#include "imgui.h"
#include "mtao/geometry/bounding_box.hpp"
#include "mtao/geometry/mesh/boundary_facets.h"
#include "mtao/geometry/mesh/read_obj.hpp"
#include "mtao/opengl/Window.h"
#include "mtao/opengl/drawables.h"

using namespace mtao::opengl;

class MeshViewer : public mtao::opengl::Window3 {
  public:
    enum class Mode : int { Smoothing,
                            LSReinitialization };
    Mode mode = Mode::LSReinitialization;

    float permeability = 100.0;
    float timestep = 1000.0;
    bool animate = false;
    using Vec = mtao::VectorX<GLfloat>;
    using Vec3 = mtao::Vec3f;
    Vec data;
    Vec data_original;
    Vec dx;
    Vec dy;
    Vec signs;
    Eigen::AlignedBox<float, 3> bbox;
    Eigen::SparseMatrix<float> L;

    std::array<int, 3> N{ { 20, 20, 20 } };
    int &NI = N[0];
    int &NJ = N[1];
    int &NK = N[2];
    float scale = 1.0;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;

    MeshViewer(const Arguments &args)
      : Window3(args),
        _wireframe_shader{
            supportsGeometryShader()
              ? Magnum::Shaders::MeshVisualizer3D::Flag::Wireframe
              : Magnum::Shaders::MeshVisualizer3D::Flag{}
        } {
        bbox.min().setConstant(-1);
        bbox.max().setConstant(1);
        mv_drawable =
          new mtao::opengl::MeshDrawable<Magnum::Shaders::MeshVisualizer3D>{
              grid, _wireframe_shader, drawables()
          };
        mv_drawable->set_visibility(false);
        ;

        edge_drawable = new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat3D>{
            grid, _flat_shader, drawables()
        };
        edge_drawable->activate_triangles({});
        edge_drawable->activate_edges();
        grid.setParent(&root());
        vfield_mesh.setParent(&root());
#ifdef FLATIT
        _vf_viewer = new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat3D>{
            vfield_mesh, _flat_shader, drawables()
        };
#else
        _vf_viewer =
          new mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<3>>{
              vfield_mesh, _vf_shader, drawables()
          };
#endif
        _vf_viewer->set_visibility(false);
        update();
    }
    void update() {
        // mtao::geometry::grid::Grid3f g(std::array<int,3>{{NI,NJ,NK}});
        auto g = mtao::geometry::grid::Grid3f::from_bbox(
          bbox, std::array<int, 3>{ { NI, NJ, NK } });

        grid.set(g);
        {
            vfield_mesh.setVertexBuffer(g.vertices());
            mtao::ColVecs3f V = g.vertices();
            // V.array() -= .5;
            auto x = V.row(0).eval();
            V.row(0) = -V.row(1);
            V.row(1) = x;
            V.row(2).array() -= g.bbox().center().z();
            V.row(2) = -V.row(2).array().pow(3);
            vfield_mesh.setVFieldBuffer(V);
        }

        _vf_viewer->deactivate();
        _vf_viewer->activate_points();
    }
    void do_animation() {}
    void gui() override {
        if (ImGui::InputInt3("N", &NI)) {
            update();
        }
        if (ImGui::SliderFloat3("min", bbox.min().data(), -2, 2)) {
            bbox.min() = (bbox.min().array() < bbox.max().array())
                           .select(bbox.min(), bbox.max());
            update();
        }
        if (ImGui::SliderFloat3("max", bbox.max().data(), -2, 2)) {
            bbox.max() = (bbox.min().array() > bbox.max().array())
                           .select(bbox.min(), bbox.max());
            update();
        }
        if (ImGui::SliderFloat("scale", &scale, 0, 2)) {
            _vf_shader.setScale(scale);
        }
        if (mv_drawable) {
            mv_drawable->gui();
        }
        if (edge_drawable) {
            edge_drawable->gui();
        }

        if (_vf_viewer) {
            _vf_viewer->gui();
        }
        if (ImGui::Button("Step")) {
            do_animation();
        }
    }
    /*
    void draw() override {
        if(animate) {
            do_animation();
        }
        Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::FaceCulling);
        Window3::draw();
    }
    */
  private:
    Magnum::Shaders::MeshVisualizer3D _wireframe_shader;
    Magnum::Shaders::Flat3D _flat_shader;
    mtao::opengl::VectorFieldShader<3> _vf_shader;
    mtao::opengl::objects::Mesh<3> vfield_mesh;
    mtao::opengl::objects::Grid<3> grid;
    mtao::opengl::MeshDrawable<Magnum::Shaders::MeshVisualizer3D> *mv_drawable =
      nullptr;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat3D> *edge_drawable = nullptr;
#ifdef FLATIT
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat3D> *_vf_viewer = nullptr;
#else
    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<3>> *_vf_viewer =
      nullptr;
#endif
};

MAGNUM_APPLICATION_MAIN(MeshViewer)
