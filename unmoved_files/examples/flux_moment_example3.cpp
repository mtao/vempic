#include <fmt/format.h>

#include <vem/flux_moment_indexer3.hpp>
#include <vem/from_grid3.hpp>

void show_operators(const vem::VEMMesh3 &vem, int max_degree) {
    spdlog::info("max degree = {}", max_degree);
    if (true) {
        vem::FluxMomentIndexer3 fmi(vem, max_degree);
        const auto &fi = fmi.flux_indexer();
        const auto &mi = fmi.moment_indexer();
        const auto &Mi = fmi.monomial_indexer();
        auto fmc = fmi.get_cell(0);

        spdlog::info(
            "Flux moments: {} elements, {} total DOFs of values [{}] and "
            "degrees [{}]",
            fi.num_partitions(), fi.num_coefficients(),
            fmt::join(fi.partition_offsets(), ","),
            fmt::join(fi.degrees(), ","));
        spdlog::info(
            "cell moments: {} elements, {} total DOFs of values [{}] and "
            "degrees [{}]",
            mi.num_partitions(), mi.num_coefficients(),
            fmt::join(mi.partition_offsets(), ","),
            fmt::join(mi.degrees(), ","));
        spdlog::info(
            "cell monomials: {} elements, {} total DOFs of values [{}] and "
            "degrees [{}]",
            Mi.num_partitions(), Mi.num_coefficients(),
            fmt::join(Mi.partition_offsets(), ","),
            fmt::join(Mi.degrees(), ","));

        // std::cout << fmc.sample_monomial_dirichlet_grammian() << std::endl;
        // std::cout << fmc.dirichlet_projector() << std::endl;
        // auto Pi = fmc.dirichlet_projector();
        // auto SG = fmc.monomial_dirichlet_grammian();

        if (true) {
            // auto MG = fmc.monomial_dirichlet_grammian();
            // auto SMG = fmc.sample_monomial_dirichlet_grammian();
            // std::cout << "MG\n" << MG << std::endl;
            // std::cout << "SMG\n" << SMG << std::endl;
            auto E = fmc.monomial_evaluation();
             auto Pi = fmc.dirichlet_projector();
             auto Pi0 = fmc.l2_projector();
            // auto G = fmc.monomial_dirichlet_grammian();
            auto RG = fmc.regularized_monomial_dirichlet_grammian();
            auto D = fmc.monomial_evaluation();
            auto B =
                fmc.sample_monomial_dirichlet_grammian().transpose().eval();
            // std::cout << "G\n" << G << std::endl;
            std::cout << "D\n" << D << std::endl;
            std::cout << "B\n" << B << std::endl;
            std::cout << "RG\n" << RG << std::endl;
            std::cout << "BD\n" << B * D << std::endl;

            std::cout << "RG BD norm: " << (RG - B * D).norm() << std::endl;

             std::cout << "Eval:\n";
             std::cout << E << std::endl;
             std::cout << "Dirichlet" << std::endl;
             std::cout << Pi << std::endl;
             std::cout << "L2" << std::endl;
             std::cout << Pi0 << std::endl;

             std::cout << "Dirichlet eval" << std::endl;
             std::cout << Pi * E << std::endl;
             std::cout << "L2 eval" << std::endl;
             std::cout << Pi0 * E << std::endl;
            Eigen::AlignedBox<double, 2> face_bbox;
            face_bbox.min().setConstant(-.5);
            face_bbox.max().setConstant(.5);
            /*
            for (auto &&[fidx, sgn] : fmc.faces()) {
                spdlog::warn("Working through face {}", fidx);
                auto face_integrals =
                    fmc.monomial_face_integrals(max_degree - 1, fidx);
                //(
                //    fidx, max_degree - 1, max_degree);
                std::cout << "integrals: " << face_integrals.transpose()
                          << std::endl;
                const auto &F = vem.face_frames.at(fidx);
                const auto &FC = vem.FC.col(fidx);
                for (int k = 0; k < fmc.flux_size(fidx); ++k) {
                    int N = 100000;
                    double val = 0;
                    auto proj_poly = fmc.project_monomial_to_boundary(fidx, k);
                    for (int u = 0; u < N; ++u) {
                        auto uv = face_bbox.sample();
                        auto p = F * uv + FC;
                        double tval =
                            fmc.flux_indexer().monomial_evaluation(fidx, k, p);

                        double new_mval = 0;
                        for (int j = 0; j < proj_poly.size(); ++j) {
                            new_mval += proj_poly(j) *
                                        fmc.flux_indexer().monomial_evaluation(
                                            fidx, j, p);
                        }
                        double mval =
                            fmc.monomial_indexer().evaluate_monomial(0, k, p);
                        if (std::abs(new_mval - mval) > 1e-8) {
                            spdlog::error(
                                "Failed to evaluate polynomial {} on boundary "
                                "({} != {})",
                                k, mval, new_mval);
                        }
                        val += tval;  // * new_mval;
                        // if (j == 4) {
                        //    spdlog::info("Val += {} ({} = {})", tval, mval,
                        //    new_mval);
                        //}
                    }
                    val *= vem.surface_area(fidx) / N;
                    spdlog::info("Quadrature check: R({}) = {} ~ {}", k,
                                 face_integrals(k), val);
                }
            }
            */
        }
        if (false) {
            if (max_degree >= 2) {
                auto E = fmc.monomial_evaluation();
                mtao::MatXd QQ = E;
                E = E.bottomRows(fmc.moment_size());

                mtao::MatXd Q(E);
                Q.setZero();

                auto bb = fmc.bounding_box();
                int N = 100000;
                mtao::ColVecs3d V(3, N);

                for (int u = 0; u < N; ++u) {
                    mtao::Vec3d p = bb.sample();
                    V.col(u) = p;
                    mtao::VecXd eval =
                        fmc.evaluate_monomials_by_size(Q.cols(), p);
                    Q += eval.head(fmc.moment_size()) * eval.transpose();
                }
                Q /= N;
                std::vector<std::set<int>> ci(1);
                for (int j = 0; j < N; ++j) {
                    ci[0].emplace(j);
                }
                for (size_t mon_idx = 0; mon_idx < fmc.monomial_size();
                     ++mon_idx) {
                    QQ.col(mon_idx) =
                        fmi.coefficients_from_point_sample_function(
                            [&](const mtao::Vec3d &p) -> double {
                                return fmc.monomial_indexer().evaluate_monomial(
                                    0, mon_idx, p);
                            },
                            V, ci);
                }
                QQ = QQ.bottomRows(fmc.moment_size());

                std::cout << "Analytical moment evaluation:\n"
                          << E << std::endl;
                std::cout << "Quadrature moment evaluation:\n"
                          << Q << std::endl;

                std::cout << "Quadrature Func moment evaluation:\n"
                          << QQ << std::endl;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    Eigen::AlignedBox<double, 3> bbox;
    bbox.min().setConstant(0);
    bbox.max().setConstant(1);
    auto vem = vem::from_grid(bbox, 2, 2, 2);
    show_operators(vem, 1);
    show_operators(vem, 2);
    // show_operators(vem, 3);
    // show_operators(vem, 4);

    return 0;
}
