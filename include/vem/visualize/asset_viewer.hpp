#pragma once
#include <mtao/opengl/objects/types.h>

#include <set>
#include <string>

#include "vem/visualize/inventory.hpp"

namespace vem::visualize {

class AssetViewer : public mtao::opengl::objects::Object2D {
   public:
    AssetViewer();
    AssetViewer(const Inventory& inv, const std::string& name);
    bool load(const Inventory& inv, const std::string& name);

    virtual std::string viewer_type() const = 0;

    virtual bool valid_type(const std::string& name) const;
    virtual std::set<std::string> valid_types() const = 0;
    virtual bool valid_storage_type(const std::string& name) const;
    virtual std::set<std::string> valid_storage_types() const = 0;

   protected:
    virtual bool load_implementation(const Inventory& inv,
                                     const std::string& name) const = 0;
};

}  // namespace vem::visualize
