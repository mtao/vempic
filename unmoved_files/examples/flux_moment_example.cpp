#include <fmt/format.h>

#include <vem/flux_moment_indexer.hpp>
#include <vem/from_polygons.hpp>
#include <vem/point_moment_indexer.hpp>

void show_operators(const vem::VEMMesh2 &vem, int max_degree) {
    spdlog::info("max degree = {}", max_degree);
    if (true) {
        vem::FluxMomentIndexer fmi(vem, max_degree);
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

        {
            // auto MG = fmc.monomial_dirichlet_grammian();
            // auto SMG = fmc.sample_monomial_dirichlet_grammian();
            // std::cout << "MG\n" << MG << std::endl;
            // std::cout << "SMG\n" << SMG << std::endl;
            // auto E = fmc.monomial_evaluation();
            // auto Pi = fmc.dirichlet_projector();
            // auto Pi0 = fmc.l2_projector();
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
            // std::cout << "Eval:\n";
            // std::cout << E << std::endl;
            // std::cout << "Dirichlet" << std::endl;
            // std::cout << Pi << std::endl;
            // std::cout << "L2" << std::endl;
            // std::cout << Pi0 << std::endl;

            // std::cout << "Dirichlet eval" << std::endl;
            // std::cout << Pi * E << std::endl;
            // std::cout << "L2 eval" << std::endl;
            // std::cout << Pi0 * E << std::endl;
            for (auto &&[eidx, sgn] : fmc.edges()) {
                const auto e = vem.E.col(eidx);
                const auto a = vem.V.col(e(0));
                const auto b = vem.V.col(e(1));
                for (int k = 0; k < fmc.monomial_size(); ++k) {
                    int N = 10;
                    double val = 0;
                    auto proj_poly = fmc.project_monomial_to_boundary(eidx, k);
                    for (int u = 0; u < N; ++u) {
                        double t = double(u) / N;
                        mtao::Vec2d p = (1 - t) * a + t * b;
                        double tval =
                            fmc.flux_indexer().monomial_evaluation(eidx, k, p);

                        double new_mval = 0;
                        for (int j = 0; j < proj_poly.size(); ++j) {
                            new_mval += proj_poly(j) *
                                        fmc.flux_indexer().monomial_evaluation(
                                            eidx, j, p);
                        }
                        double mval =
                            fmc.monomial_indexer().evaluate_monomial(0, k, p);
                        if (std::abs(new_mval - mval) > 1e-8) {
                            spdlog::error(
                                "Failed to evaluate polynomial {} on boundary "
                                "({} != {})",
                                k, mval, new_mval);
                        }
                        // val += tval;  // * new_mval;
                        // if (j == 4) {
                        //    spdlog::info("Val += {} ({} = {})", tval, mval,
                        //    new_mval);
                        //}
                    }
                    // val *= edge_length(edge_index) / N;
                    // spdlog::info("Quadrature check: R({}) = {} ~ {}", j,
                    //             edge_integrals(j), val);
                }
            }
            if (max_degree >= 2) {
                auto E = fmc.monomial_evaluation();
                mtao::MatXd QQ = E;
                E = E.bottomRows(fmc.moment_size());

                mtao::MatXd Q(E);
                Q.setZero();

                auto bb = fmc.bounding_box();
                int N = 10;
                mtao::ColVecs2d V(2, N);

                for (int u = 0; u < N; ++u) {
                    mtao::Vec2d p = bb.sample();
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
                            [&](const mtao::Vec2d &p) -> double {
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

    if (false) {
        spdlog::error("PointMomentIndexer stuff");
        vem::PointMomentIndexer pmi(vem, max_degree);

        auto fmc = pmi.get_cell(0);
        // std::cout << c.dirichlet_projector() * c.monomial_evaluation()
        //          << std::endl;
        // auto Pi = c.dirichlet_projector();
        // auto SG = c.monomial_dirichlet_grammian();

        // std::cout << Pi.transpose() * SG * Pi << std::endl;
        {
            auto E = fmc.monomial_evaluation();
            auto Pi = fmc.dirichlet_projector();
            auto Pi0 = fmc.l2_projector();
            auto G = fmc.monomial_dirichlet_grammian();
            auto RG = fmc.regularized_monomial_dirichlet_grammian();
            auto D = fmc.monomial_evaluation();
            auto B =
                fmc.sample_monomial_dirichlet_grammian().transpose().eval();

            auto MG = fmc.monomial_l2_grammian();
            auto SMG = fmc.sample_monomial_l2_grammian();
            std::cout << "Monomial Grammian\n" << MG << std::endl;
            // std::cout << "G\n" << G << std::endl;
            std::cout << "D\n" << D << std::endl;
            std::cout << "B\n" << B << std::endl;
            std::cout << "RG\n" << RG << std::endl;
            std::cout << "BD\n" << B * D << std::endl;
            std::cout << "RG BD norm: " << (RG - B * D).norm() << std::endl;
            // std::cout << "Eval:\n";
            // std::cout << E << std::endl;
            // std::cout << "Dirichlet" << std::endl;
            // std::cout << Pi << std::endl;
            // std::cout << "L2" << std::endl;
            // std::cout << Pi0 << std::endl;

            std::cout << "Dirichlet eval" << std::endl;
            std::cout << Pi * E << std::endl;
            // std::cout << "L2 eval" << std::endl;
            // std::cout << Pi0 * E << std::endl;
        }
        // std::cout << c.l2_projector() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    mtao::vector<mtao::ColVecs2d> polys;
    auto &P = polys.emplace_back();
    P.resize(2, 4);
    P.col(0) << 0, 0;
    P.col(1) << 1, 0;
    P.col(2) << 1, 1;
    P.col(3) << 0, 1;

    auto vem = vem::from_polygons(polys);
    std::cout << vem.E << std::endl;
    show_operators(vem, 1);
    show_operators(vem, 2);
    show_operators(vem, 3);
    // show_operators(vem, 4);

    return 0;
}
