#include <fmt/format.h>

#include <iostream>
#include <mtao/iterator/enumerate.hpp>
#include <vem/from_mandoline.hpp>

int main(int argc, char *argv[]) {
    Eigen::AlignedBox<double, 2> bb;
    bb.min().setConstant(-1);
    bb.max().setConstant(1);

    mtao::ColVecs2i E(2, 2);
    mtao::ColVecs2d V(2, 3);

    V.col(0) << 0, .3;
    V.col(1) << 0.1, 0.1;
    V.col(2) << 0.3, 0;
    E.col(0) << 0, 1;
    E.col(1) << 1, 2;

    auto vem = vem::from_mandoline(bb,5,5, V, E, true);


    return 0;
}
