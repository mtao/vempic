#pragma once
#include <Magnum/Shaders/Flat.h>

#include <mtao/opengl/objects/partitioned_polynomial_shaded_mesh.hpp>

#include "vem/mesh.hpp"
#include "vem/monomial_basis_indexer.hpp"

namespace vem::visualize {

class VEM2ScalarFieldViewer
    : public mtao::opengl::objects::PartitionedPolynomialShadedMesh<2> {
   public:
    using Base = mtao::opengl::objects::PartitionedPolynomialShadedMesh<2>;
    VEM2ScalarFieldViewer(const VEMMesh2 &mesh,
                          std::shared_ptr<const MonomialBasisIndexer> mi,
                          Base::DrawableGroup *group = nullptr);

    VEM2ScalarFieldViewer(Base::DrawableGroup *group = nullptr);
    using TransMat = Base::TransMat;
    using Camera = Base::Camera;
    enum ViewMode : char { Points = 1, Field = 1 << 1 };

    void set_mesh(const std::shared_ptr<const VEMMesh2> &);
    void set_mesh(const VEMMesh2 &);
    void set_monomial_indexer(
        std::shared_ptr<const MonomialBasisIndexer> indexer,
        bool shortcut_asset_update = false);
    void set_coefficients(const mtao::VecXd &coeffs);
    void set_scales(const mtao::VecXd &scales);
    void set_degrees(const mtao::VecXi &scales);
    bool gui();

    void draw(const TransMat &transformationMatrix, Camera &camera) override;
    void setPointValues(const mtao::VecXd &F);

    std::set<int> active_cells = {};

    void update_mesh_visualization();
    void set_show_field(bool value);

   private:
    char _view_mode = ViewMode::Field;
    float _point_size = 5.f;
    float _point_outline_size = 2.f;
    mtao::VecXd _point_data;
    std::shared_ptr<const MonomialBasisIndexer> _monomial_indexer;
    Magnum::Shaders::VertexColor2D _vertex_color_shader;
    Magnum::Shaders::Flat2D _black_shader;

    void updatePointValues();
};

}  // namespace vem::visualize
