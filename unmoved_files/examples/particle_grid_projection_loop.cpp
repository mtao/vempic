#include "vem/point_moment_indexer.hpp"
#include <vem/fluidsim_2d/fluidvem2.hpp>
#include <vem/fluidsim_2d/sim.hpp>
#include <vem/from_grid.hpp>
#include <vem/poisson_2d/example_constraints.hpp>

auto make_mesh(int N) {
    Eigen::AlignedBox<double, 2> bb;
    bb.min().setConstant(0);
    bb.max().setConstant(1);
    auto mesh = vem::from_grid(bb, N, N);

    return mesh;
}
auto make_sim(const vem::Mesh2& mesh, int degree) {
    return vem::fluidsim_2d::Sim(mesh,degree);
}

void get_eigen_decomposition(int N, int D) {
    spdlog::info("N = {}, D = {}", N,D);
    auto mesh = make_mesh(N);
    auto sim = make_sim(mesh,D);

}

void test_projection_operators(int N, int D) {
    auto mesh = make_mesh(N);
    vem::fluidsim_2d::FluidVEM2 fvem(mesh, D);
}
int main(int argc, char *argv[]) {  // test_projected_divergence(3);
    // test_projection_operators(2, 1);
    test_projection_operators(2, 2);
    //test_projection_operators(2, 1);
}
