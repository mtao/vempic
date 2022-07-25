#pragma once
#include <Magnum/SceneGraph/SceneGraph.h>
#include <mtao/opengl/drawables.h>
#include <mtao/opengl/objects/mesh.h>
#include <mtao/opengl/objects/types.h>

#include <map>
#include <mtao/python/load_python_function.hpp>
#include <mtao/types.hpp>
#include <vem/mesh.hpp>

#include "vem/poisson_2d/constraints.hpp"
#include "vem/visualize/vem_scalar_field_viewer.hpp"

namespace vem::serialization {
class Inventory;
}
namespace vem::poisson_2d {

struct ScalarConstraintsGui : public ScalarConstraints,
                              public mtao::opengl::Object2D,
                              public Magnum::SceneGraph::Drawable2D {
    using TransMat = Magnum::Math::Matrix3<float>;
    using Camera = Magnum::SceneGraph::Camera2D;
    // designed to take two float parameters from a
    // ShaderData<PolynomialScalarFieldShader> object
    ScalarConstraintsGui(const visualize::VEM2ScalarFieldViewer &viewer,
                         Magnum::SceneGraph::DrawableGroup2D *group);
    using ScalarConstraints::clear;
    bool gui(const VEMMesh2 &mesh);
    void update_mesh(const VEMMesh2 &mesh);

    // assignment just overwrites the underlying constraint object
    ScalarConstraintsGui &operator=(const ScalarConstraints &constraints);
    void draw(const TransMat &transformationMatrix, Camera &camera) override;

   private:
    void update_vertex(const VEMMesh2 &mesh);
    void update_edge(const VEMMesh2 &mesh);

    mtao::opengl::objects::Mesh<2> _vertex_mesh, _edge_mesh;
    Magnum::Shaders::Flat2D _flat_shader;
    int _vertex_index = -1;
    float _vertex_value = 0;
    Magnum::Color4 _vertex_color;
    int _edge_index = -1;
    float _edge_value = 0;
    Magnum::Color4 _edge_color;

    float poly_constant = 0;
    mtao::Vec2f poly_linear = mtao::Vec2f(1.f, 0.f);

    // maps (point,time) -> (vector,bool)
    std::shared_ptr<mtao::python::PythonFunction> boundary_conditions;
    std::string boundary_condition_function;
    void update_boundary_condition_from_func(const VEMMesh2 &mesh);

    serialization::Inventory *inventory = nullptr;

    const visualize::VEM2ScalarFieldViewer &_viewer;
};
}  // namespace vem::poisson_2d
