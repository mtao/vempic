#include "vem/utils/monomial_coefficient_projection.hpp"

#include <spdlog/spdlog.h>

namespace vem::utils {

Eigen::SparseMatrix<double> monomial_coefficient_projection(
    const MonomialBasisIndexer& from, const MonomialBasisIndexer& to) {
    std::vector<Eigen::Triplet<double>> trips;
    Eigen::SparseMatrix<double> A(to.num_coefficients(),
                                  from.num_coefficients());
    trips.reserve(std::min(A.rows(), A.cols()));
    if (from.num_partitions() != to.num_partitions()) {
        spdlog::error(
            "monomial coefficient projection encountered difference cell "
            "sizes! make sure you're pssing compatible monomial tools with "
            "compatible cells");  // this doesn't guarantee an error didn't
                                  // happen, but i dont think i have operator==
                                  // on the VEM mesh itself :
    }

    // spdlog::info("Number of partitions: {} {}", from.num_partitions(),
    //             to.num_partitions());
    for (size_t j = 0; j < from.num_partitions(); ++j) {
        auto f = from.coefficient_indices(j);
        auto t = to.coefficient_indices(j);
        auto [tstart, tend] = to.coefficient_range(j);
        auto [fstart, fend] = from.coefficient_range(j);
        int count = std::min(tend - tstart, fend - fstart);

        // spdlog::info("{}=>{} {}=>{}", tstart, tend, fstart, fend);
        for (int j = 0; j < count; ++j) {
            trips.emplace_back(tstart + j, fstart + j, 1);
        }
    }
    // std::cout << "Triplet count: " << trips.size() << std::endl;
    A.setFromTriplets(trips.begin(), trips.end());
    return A;
}
}  // namespace vem::utils

