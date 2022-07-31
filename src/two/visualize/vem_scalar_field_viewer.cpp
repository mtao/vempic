#include "vem/two/visualize/vem_scalar_field_viewer.hpp"

#include <Magnum/GL/Renderer.h>

#include <mtao/eigen/stack.hpp>

namespace vem::two::visualize {

VEM2ScalarFieldViewer::VEM2ScalarFieldViewer(
    const VEMMesh2 &mesh, std::shared_ptr<const MonomialBasisIndexer> mi,
    Base::DrawableGroup *group)
    : Base(group), _monomial_indexer(mi) {
    set_mesh(mesh);
    _black_shader.setColor(Magnum::Math::Color4<float>(0., 0., 0., 1.));
}

VEM2ScalarFieldViewer::VEM2ScalarFieldViewer(Base::DrawableGroup *group)
    : Base(group) {
    _black_shader.setColor(Magnum::Math::Color4<float>(0., 0., 0., 1.));
}
void VEM2ScalarFieldViewer::set_mesh(
    const std::shared_ptr<const VEMMesh2> &mesh) {
    set_mesh(*mesh);
}
void VEM2ScalarFieldViewer::set_mesh(const VEMMesh2 &mesh) {
    auto faces = mesh.triangulated_faces();

    std::vector<size_t> sizes;
    sizes.reserve(faces.size());
    std::transform(faces.begin(), faces.end(), std::back_inserter(sizes),
                   [](const mtao::ColVecs3i &F) { return F.cols(); });

    std::vector<int> offsets(faces.size() + 1);
    offsets[0] = 0;
    std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1,
                     std::plus<size_t>{});
    auto F = mtao::eigen::hstack_iter(faces.begin(), faces.end());

    setTriangleBuffer(mesh.V.topRows<2>().cast<float>(),
                      F.cast<unsigned int>());
    set_offsets(offsets);
    auto &coeffs = coefficients();

    for (auto &&[index, pc] : mtao::iterator::enumerate(coeffs)) {
        auto c = mesh.C.col(index);
        auto &cc = pc.center;
        cc[0] = c.x();
        cc[1] = c.y();
    }
}
void VEM2ScalarFieldViewer::update_mesh_visualization() {}

void VEM2ScalarFieldViewer::set_scales(const mtao::VecXd &scales) {
    auto &coeffs = coefficients();
    for (auto &&[pc,scale] : mtao::iterator::zip(coeffs,scales)) {
        auto &cc = pc.scale;
        cc = scale;
    }
}
void VEM2ScalarFieldViewer::set_degrees(const mtao::VecXi &degrees) {
    auto &coeffs = coefficients();
    for (auto &&[pc,degree] : mtao::iterator::zip(coeffs,degrees)) {
        auto &cc = pc.degree;
        cc = degree;
    }
}
void VEM2ScalarFieldViewer::set_coefficients(const mtao::VecXd &coeffs) {
    if (!_monomial_indexer) {
        spdlog::warn(
            "VEM2ScalarFieldViewer cant set coefficients without a monomial "
            "indexer");
        return;
    }
    if (coeffs.size() != _monomial_indexer->num_coefficients()) {
        spdlog::warn(
            "Different #coefficients were passed to visualize polynomials than "
            "necessary, got {}, expected {}",
            coeffs.size(), _monomial_indexer->num_coefficients());
        return;
    }
    for (auto &&[index, pc] : mtao::iterator::enumerate(coefficients())) {
        auto [start, end] = _monomial_indexer->coefficient_range(index);
        auto B = coeffs.segment(start, end - start);
        pc.degree = _monomial_indexer->degree(index);
        pc.zero();

        switch (pc.degree) {
            case 3:
                // x^3
                pc.cubic[0][0][0] = B(6);
                // x^2 y
                pc.cubic[0][0][1] = B(7);
                pc.cubic[0][1][0] = 0;
                // x y^2
                pc.cubic[0][1][1] = B(8);

                pc.cubic[1][0][0] = 0;
                pc.cubic[1][0][1] = 0;
                pc.cubic[1][1][0] = 0;
                // y^3
                pc.cubic[1][1][1] = B(9);
            case 2:
                // x^2
                pc.quadratic[0][0] = B(3);
                // x y
                pc.quadratic[0][1] = B(4);
                pc.quadratic[1][0] = 0;
                // y^2
                pc.quadratic[1][1] = B(5);

            case 1:
                // x
                pc.linear[0] = B(1);
                // y
                pc.linear[1] = B(2);
            case 0:
                pc.constant = B(0);
            default:
                continue;
        }
    }
}
void VEM2ScalarFieldViewer::set_monomial_indexer(
    std::shared_ptr<const MonomialBasisIndexer> indexer,
    bool shortcut_asset_update) {
    if (indexer == nullptr) {
        spdlog::info("empty monomial basis indexer passed over");
        return;
    }
    // a shortcut that might really bite me later
    if (shortcut_asset_update && indexer == _monomial_indexer) {
        return;
    }
    _monomial_indexer = indexer;

    set_scales(mtao::eigen::stl2eigen(_monomial_indexer->diameters()));

    set_degrees(
        mtao::eigen::stl2eigen(_monomial_indexer->degrees()).cast<int>());
}

void VEM2ScalarFieldViewer::draw(const TransMat &transformationMatrix,
                                 Camera &camera) {
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::FaceCulling);
    if (_view_mode & static_cast<char>(ViewMode::Field)) {
        Base::shader().setColormapScale(Base::shader_data().colormap_scale);
        Base::shader().setColormapShift(Base::shader_data().colormap_shift);

        using M = mtao::opengl::objects::Mesh<2>;

        M::addVertexBuffer(
            M::vertex_buffer, 0,
            mtao::opengl::PolynomialScalarFieldShader<2>::Position{});

        // MeshType::setIndexBuffer(MeshType::triangle_index_buffer, 0,
        //                         MeshType::triangle_indexType, 0, 0);
        MeshType::setPrimitive(Magnum::GL::MeshPrimitive::Triangles);

        Base::shader().setTransformationProjectionMatrix(
            camera.projectionMatrix() * transformationMatrix);

        // for (auto&& [index,coeffs] :
        // mtao::iterator::enumerate(_coefficients)) {
        for (auto &&[idx, view, coeffs] :
             mtao::iterator::enumerate(views(), coefficients())) {
            if (active_cells.empty() || active_cells.contains(idx)) {
                Base::shader().setPolynomialCoefficients(coeffs);
                Base::shader().draw(view);
            } else {
            }
            // Base::shader().draw(view);
        }
    }

    using M = mtao::opengl::objects::Mesh<2>;
    if (_view_mode & static_cast<char>(ViewMode::Points) &&
        M::color_buffer.size() > 0) {
        MeshType::setPrimitive(Magnum::GL::MeshPrimitive::Points);

        Magnum::GL::Renderer::setPointSize(_point_size +
                                           2 * _point_outline_size);
        _black_shader.setTransformationProjectionMatrix(
            camera.projectionMatrix() * transformationMatrix);
        _black_shader.draw(*this);
        Magnum::GL::Renderer::setPointSize(_point_size);
        _vertex_color_shader.setTransformationProjectionMatrix(
            camera.projectionMatrix() * transformationMatrix);
        M::addVertexBuffer(M::vertex_buffer, 0,
                           Magnum::Shaders::VertexColor2D::Position{});
        M::addVertexBuffer(M::color_buffer, 0,
                           Magnum::Shaders::VertexColor2D::Color4{});
        _vertex_color_shader.draw(*this);
    }
}
void VEM2ScalarFieldViewer::setPointValues(const mtao::VecXd &F) {
    _point_data = F;
    updatePointValues();
}

void VEM2ScalarFieldViewer::updatePointValues() {
    mtao::ColVecs4f C(4, _point_data.size());
    for (int j = 0; j < _point_data.size(); ++j) {
        C.col(j) = Base::get_color(_point_data(j));
    }

    Base::setColorBuffer(C);
}

bool VEM2ScalarFieldViewer::gui() {
    bool points = _view_mode & static_cast<char>(ViewMode::Points);
    bool field = _view_mode & static_cast<char>(ViewMode::Field);

    if (ImGui::InputFloat("Point size", &_point_size)) {
        return true;
    }
    if (ImGui::InputFloat("Point outline size", &_point_outline_size)) {
        return true;
    }
    if (ImGui::Checkbox("View Points", &points)) {
        _view_mode &= static_cast<char>(ViewMode::Field);
        _view_mode |= points ? static_cast<char>(ViewMode::Points) : 0;
        return true;
    }
    if (ImGui::Checkbox("View Field", &field)) {
        _view_mode &= static_cast<char>(ViewMode::Points);
        _view_mode |= field ? static_cast<char>(ViewMode::Field) : 0;
        return true;
    }

    if (Base::gui()) {
        updatePointValues();
        return true;
    }
    return false;
}
void VEM2ScalarFieldViewer::set_show_field(bool value) {
    // activate everything but field
    _view_mode &= static_cast<char>(ViewMode::Points);
    // set field
    _view_mode |= value ? static_cast<char>(ViewMode::Field) : 0;
}

}  // namespace vem::visualize
