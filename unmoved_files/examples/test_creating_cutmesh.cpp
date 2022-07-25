#include <vem/creator2.hpp>

#include <mtao/geometry/bounding_box.hpp>

int main(int argc, char * argv[]) {
    


    vem::VEMMesh2Creator creator;
    creator.mesh_filename = argv[1];
    creator.load_boundary_mesh();

    auto&& [V,E] = *creator._held_boundary_mesh;
    std::cout << "V\n" << V << std::endl;
    std::cout << "E\n" << E << std::endl;
    for(int j = 0; j < E.cols(); ++j) {
        std::cout << V.col(E.col(j)(0)).transpose() << std::endl;
    }
    auto bb = mtao::geometry::bounding_box(V);
    bb.min().array() -= 1;
    bb.max().array() -= 1;


    creator.grid_mesh_bbox = bb.cast<float>();

    creator.make_mandoline_mesh(true);
    return 0;


}
