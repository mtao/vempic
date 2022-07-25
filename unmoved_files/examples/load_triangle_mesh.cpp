#include <fmt/format.h>

#include <iostream>
#include <mtao/geometry/mesh/read_obj.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <vem/from_simplicial_matrices.hpp>
#include <vem/poisson_2d/poisson_vem.hpp>
#include <vem/point_sample_indexer.hpp>

int main(int argc, char *argv[]) {
    auto [V, F] = mtao::geometry::mesh::read_objD(argv[1]);
    if (F.minCoeff() < 0) {
        F.array() += 1;
    }
    auto vem = vem::from_triangle_mesh(V.topRows<2>(), F);

    std::cout << "Vertices: \n"
              << vem.V.transpose() << std::endl;
    /*
    std::cout << "Centers: \n" << vem.C.transpose() << std::endl;
    std::cout << "Boundary: " << std::endl;
    for (auto&& [cell_index, cell_map] :
         mtao::iterator::enumerate(vem.face_boundary_map)) {
        fmt::print("cell_index({})", cell_index);
        for (auto&& [c, sgn] : cell_map) {
            fmt::print("{}=>{} ", c, sgn ? -1 : 1);
        }
        fmt::print("\n");
    }
    */

    vem::PointSampleIndexer point_sample(vem, 1);
    for (int j = 0; j < point_sample.num_coefficients(); ++j) {
        std::cout << point_sample.get_position(j).transpose() << std::endl;
    }
    vem::poisson_2d::PoissonVEM2 pois(vem, 1);

    std::cout << pois.stiffness_matrix() << std::endl;

    return 0;
}
