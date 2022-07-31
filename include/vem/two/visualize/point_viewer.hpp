#pragma once
#include <mtao/opengl/drawables.h>

#include "vem/visualize/asset_viewer.hpp"

namespace vem::visualize {
class PointViewer : public AssetViewer,
                    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>* poi {
    PointViewer();
    PointViewer(const Inventory& inv, const std::string& name);

    std::string viewer_type() const override { return "PointViewer"; }

    std::set<std::string> valid_types() const override;
    std::set<std::string> valid_storage_types() const override;
    std::string current_asset() const override;
    std::set<std::string> valid_types() const override;
    std::set<std::string> valid_storage_types() const override;

    void update();

   protected:
    bool load_implementation(const Inventory& inv,
                             const std::string& name) const override;

   private:
    mtao::opengl::objects::Mesh<2> _mesh;
    std::map<std::string, mtao::ColVecs2f> point_data;
    std::optional<std::string> current_asset_name;
};
}  // namespace vem::visualize
