#include <tbb/parallel_for.h>

#include <mtao/geometry/bounding_box.hpp>
#include <mtao/geometry/point_cloud/bridson_poisson_disk_sampling.hpp>
#include <vem/cell3.hpp>
#include <vem/creator3.hpp>
#include <vem/flux_moment_cell3.hpp>
#include <vem/flux_moment_indexer3.hpp>
#include <vem/from_mandoline3.hpp>
#include <vem/monomial_basis_indexer_new.hpp>
#include <vem/monomial_cell_integrals.hpp>
#include <vem/polynomial_gradient.hpp>
#include <vem/utils/face_boundary_facets.hpp>

#include "mtao/eigen/sparse_block_diagonal_repmats.hpp"
#include "vem/mesh.hpp"

namespace {

double integrate(auto&& f, auto&& bbox) {
    double val = 0;
    int N = 30;
    using Vec = typename std::decay_t<decltype(bbox)>::VectorType;
    Vec C = bbox.center();
    double rad = std::pow<double>(bbox.sizes().maxCoeff(), 1 / 3.) / N;
    auto P =
        mtao::geometry::point_cloud::bridson_poisson_disk_sampling(bbox, rad);

    for (int m = 0; m < P.cols(); ++m) {
        auto s = P.col(m);
        Vec v;
        Vec o = 2 * C - s;
        for (int j = 0; j < 2; ++j) {
            v(0) = (j == 0) ? s(0) : o(0);
            for (int k = 0; k < 2; ++k) {
                v(1) = (k == 0) ? s(1) : o(1);
                if constexpr (Vec::RowsAtCompileTime == 3) {
                    for (int l = 0; l < 2; ++l) {
                        v(2) = (l == 0) ? s(2) : o(2);
                        val += f(v);
                    }
                } else {
                    val += f(v);
                }
            }
        }
    }
    val /= 8 * P.cols();
    val *= bbox.sizes().prod();
    return val;
}

double average_inside(auto&& f, auto&& inside_predicate, auto&& bbox) {
    double val = 0;
    int N = 100;
    using Vec = typename std::decay_t<decltype(bbox)>::VectorType;
    Vec C = bbox.center();
    double rad = std::pow<double>(bbox.sizes().maxCoeff(), 1 / 3.) / N;
    auto P =
        mtao::geometry::point_cloud::bridson_poisson_disk_sampling(bbox, rad);

    int valid_count = 0;
    for (int m = 0; m < P.cols(); ++m) {
        auto s = P.col(m);
        Vec v;
        Vec o = 2 * C - s;
        for (int j = 0; j < 2; ++j) {
            v(0) = (j == 0) ? s(0) : o(0);
            for (int k = 0; k < 2; ++k) {
                v(1) = (k == 0) ? s(1) : o(1);
                if constexpr (Vec::RowsAtCompileTime == 3) {
                    for (int l = 0; l < 2; ++l) {
                        v(2) = (l == 0) ? s(2) : o(2);
                        if (inside_predicate(v)) {
                            val += f(v);
                            valid_count++;
                        }
                    }
                } else {
                    if (inside_predicate(v)) {
                        val += f(v);
                        valid_count++;
                    }
                }
            }
        }
    }
    val /= valid_count;
    return val;
}
}  // namespace
//#include "vem/from_mandoline3.hpp"

void test(const vem::VEMMesh3& mesh) {
    auto clean = [](auto&& M) {
        auto D = Eigen::MatrixXd(M);
        D = (D.array().abs() > 1e-8).select(D, 0);
        return D;
    };
    vem::FluxMomentIndexer3 fmi(mesh, 2);
    for (int j = 0; j < mesh.cell_count(); ++j) {
        int grade = mesh.grade(j);
        spdlog::error("J: {} (grade = {})", j, grade);
        // if (grade == 0) {
        //} else {
        //    continue;
        //}
        auto c = fmi.get_cell(j);
        const auto& fmc = c;

        spdlog::info("ER: {}", fmc.face_count());
        if (fmc.face_count() > 12) {
            continue;
        }
        auto E = fmc.monomial_evaluation();

        const auto G = fmc.monomial_dirichlet_grammian();
        const auto RG = fmc.regularized_monomial_dirichlet_grammian();
        const auto D = fmc.monomial_evaluation();
        const auto B =
            fmc.sample_monomial_dirichlet_grammian().transpose().eval();
        const auto bbox = c.bounding_box();
        std::cout << "G\n" << clean(G) << std::endl;
        std::cout << "D\n" << clean(D) << std::endl;
        std::cout << "B\n" << clean(B) << std::endl;
        //// std::cout << "RG\n" << clean(RG) << std::endl;
        // std::cout << "BD\n" << clean(B * D) << std::endl;
        if (false) {
            auto R = G;
            R.setZero();
            std::cout << " Monomial dirichlet grammian" << std::endl;
            std::cout << G << std::endl;
            tbb::parallel_for(int(0), int(G.rows()), [&](int j) {
                auto mj = c.monomial_gradient(j);
                tbb::parallel_for(int(0), int(G.cols()), [&](int k) {
                    auto mk = c.monomial_gradient(k);
                    R(j, k) = integrate(
                        [&](auto&& p) -> double {
                            if (c.is_inside(p)) {
                                // std::cout << mj(p).transpose() << " "
                                //          << mk(p).transpose() <<
                                //          std::endl;
                                return mj(p).dot(mk(p));
                            } else {
                                return 0;
                            }
                        },
                        bbox);
                });
            });
            std::cout << "My VMD:\n" << R << std::endl;
            std::cout << "Error (1 if good):\n";
            std::cout << ((R - G).array().abs().array() < 1e-1) << std::endl;
            return;
        }

        auto get_projected_faces =
            [&](int face_index, int count) -> Eigen::AlignedBox<double, 2> {
            Eigen::AlignedBox<double, 2> bbox;
            auto F = c.face_frame(face_index);
            auto FC = c.face_center(face_index);
            FC.colwise().normalize();
            for (auto&& vidx :
                 vem::utils::face_boundary_vertices(c.mesh(), face_index)) {
                auto v = c.mesh().V.col(vidx);
                bbox.extend(F.transpose() * (v - FC));
            }
            return bbox;
        };

        auto get_local_samples = [&](int face_index,
                                     int count) -> mtao::ColVecs3d {
            auto F = c.face_frame(face_index);
            F.colwise().normalize();

            auto T = mesh.triangulated_face(face_index);
            auto N = c.face_normal(face_index);
            for (int j = 0; j < T.cols(); ++j) {
                auto t = T.col(j);
                auto a = mesh.V.col(t(0));
                auto b = mesh.V.col(t(1));
                auto c = mesh.V.col(t(2));
                // std::cout << N.dot(b - a) << " ";
                // std::cout << N.dot(c - a) << " === " << N.transpose() << "
                // =="
                //          << (b - a).cross(c - a).transpose().normalized()
                //          << std::endl;
            }
            auto FC = c.face_center(face_index);
            // spdlog::info("N stuff");
            // std::cout << N.transpose() * F << std::endl;

            auto PV = F.transpose() * (mesh.V.colwise() - FC);

            Eigen::AlignedBox<double, 2> bbox;

            auto pc = F.transpose() * FC;

            for (auto&& vidx :
                 vem::utils::face_boundary_vertices(c.mesh(), face_index)) {
                auto v = PV.col(vidx);
                // std::cout << N.dot(mesh.V.col(vidx) - FC) << " ";
                bbox.extend(v);
            }
            // std::cout << std::endl;
            auto face_loops = mesh.face_loops(face_index);
            mtao::ColVecs3d R(3, count);
            for (int j = 0; j < count; ++j) {
                mtao::Vec2d s = bbox.sample();
                while (!face_loops.is_inside(PV, s)) {
                    s = bbox.sample();
                }
                auto p = R.col(j) = F * s + FC;
            }
            return R;
        };

        // for (auto&& [fidx, sgn] : c.faces()) {
        //    int fsize = c.flux_size(fidx);
        //    int offset = c.local_flux_index_offset(fidx);
        //    auto samples = get_local_samples(fidx, 10);
        //}

        // if (false) {
        //    for (auto&& [fidx, sgn] : c.faces()) {
        //        int fsize = c.flux_size(fidx);
        //        int offset = c.local_flux_index_offset(fidx);
        //        auto samples = get_local_samples(fidx, 10);
        //        /*
        //        for (int j = 0; j < c.monomial_size(); ++j) {
        //            auto mg = c.monomial_gradient(j);

        //            auto coeffs =
        //                c.monomial_gradient_in_face_coefficients(fidx, j);
        //            for (int k = 0; k < samples.cols(); ++k) {
        //                auto s = samples.col(k);
        //                auto mono_grad = mg(s);
        //                auto pc = c.flux_indexer().evaluate_monomials_by_size(
        //                    fidx, coeffs.cols(), s);
        //                auto face_grad = coeffs * pc;
        //                spdlog::info("{}", (mono_grad - face_grad).norm());
        //                // spdlog::info("{} {}", mono_grad, face_grad);
        //            }
        //        }
        //        */
        //    }
        //}
        if(false){
            double vol = c.volume();
            auto EN = c.face_normals();

            mtao::MatXd MyG;
            MyG.resizeLike(G);
            MyG.setZero();
            int index_offset = 0;
            {
                for (auto&& [_fidx, sgn] : c.faces()) {
                    const auto& fidx = _fidx;
                    spdlog::warn("face info: {} {}", fidx, sgn);
                    int fsize = c.flux_size(fidx);
                    int offset = c.local_flux_index_offset(fidx);
                    // auto samples = get_local_samples(fidx, 100000);
                    auto samples = fmi.weighted_face_samples(fidx, 1000)
                                        .topRows<3>()
                                        .eval();
                    mtao::Vec3d N = mtao::eigen::stl2eigen(EN[fidx]);
                    //(sgn ? -1 : 1) * mtao::eigen::stl2eigen(EN[fidx]);
                    mtao::MatXd I;
                    I.resizeLike(MyG);
                    I.setZero();
                    double sa = c.surface_area(fidx);
                    auto T = mesh.triangulated_face(fidx);
                    double sa2 = 0;
                    for (int j = 0; j < T.cols(); ++j) {
                        auto t = T.col(j);
                        auto a = mesh.V.col(t(0));
                        auto b = mesh.V.col(t(1));
                        auto c = mesh.V.col(t(2));
                        sa2 += .5 * (b - a).cross(c - a).norm();
                    }
                    std::cout << "Surface area: " << sa << std::endl;
                    std::cout << "Center: "
                              << (mesh.FC.col(fidx) - c.center()).transpose()
                              << std::endl;
                    std::cout << "Normal: " << N.transpose() << std::endl;

                    for (int row = 0; row < c.monomial_size(); ++row) {
                        auto face_coeffs =
                            c.monomial_gradient_in_face_coefficients(fidx, row);
                        for (int col = 0; col < c.monomial_size(); ++col) {
                            /*
                            tbb::parallel_for(int(0), int(c.monomial_size()),
                            [&](int row) { tbb::parallel_for( int(0),
                            int(c.monomial_size()), [&](int col) {
                                    */
                            double& my_val = I(row, col);
                            auto mi = fmc.monomial_indexer().monomial(
                                c.cell_index(), col);
                            auto mig = c.monomial_gradient(row);

                            auto mj = c.monomial_gradient(row);

                            mtao::Vec3d G;
                            G.setZero();
                            double V = 0;
                            for (int k = 0; k < samples.cols(); ++k) {
                                auto s = samples.col(k);

                                double a = mi(s);
                                V += a;

                                // auto pc =
                                //    c.flux_indexer().evaluate_monomials_by_size(
                                //        fidx, face_coeffs.cols(), s);
                                // double b = N.dot(face_coeffs * pc);
                                double b = N.dot(mj(s));
                                G += mj(s);
                                // std::cout << a << " " << b << std::endl;
                                my_val += a * b;
                            }
                            my_val /= samples.cols();
                            spdlog::info("Accumulated: ({},{}) {}", row, col,
                                         my_val);
                            std::cout << "Got a grad of " << G.transpose()
                                      << " a val " << V << std::endl;

                            my_val *= sa;
                            /*
                        });
                });
                */
                        }
                    }
                    std::cout << "Single face addition\n" << I << std::endl;
                    MyG += I;
                }
            }

            size_t mom_size = c.moment_size();
            if (mom_size > 0) {
                {
                    size_t mom_off = c.local_moment_index_offset();
                    auto L =
                        vem::polynomials::three::laplacian(c.monomial_degree());
                    std::cout << "Laplacian:\n";
                    std::cout << L << std::endl;
                    mtao::MatXd I;
                    I.resizeLike(MyG);
                    I.setZero();
                    for (int col = 0; col < c.monomial_size(); ++col) {
                        auto mcol = c.monomial(col);

                        // std::cout <<"Laplacian\n" << L << std::endl;
                        auto d = c.diameter();
                        double d2 = d * d;
                        // coefficients for \nabla m_i \nabla m_j
                        for (int o = 0; o < L.outerSize(); ++o) {
                            for (Eigen::SparseMatrix<double>::InnerIterator it(
                                     L, o);
                                 it; ++it) {
                                int intsum = it.row() + mom_off;
                                int row = it.col();

                                auto mintsum = c.monomial(it.row());
                                double nv = average_inside(
                                    [&](auto&& p) -> double {
                                        // std::cout << p.transpose() << " => "
                                        //          << it.value() << " "
                                        //          << mrow(p) << " "
                                        //          << mintsum(p) << std::endl;
                                        return it.value() * mcol(p) *
                                               mintsum(p) * d2;
                                    },
                                    [&](auto&& p) -> bool {
                                        return c.is_inside(p);
                                    },
                                    bbox);
                                std::cout << nv << std::endl;
                                std::cout << "Adding to " << row << ", " << col
                                          << " => " << nv << std::endl;
                                I(row, col) += nv;
                            }
                        }
                    }
                    std::cout << "Should be this" << std::endl;
                    mtao::MatXd M = B * D - G;
                    for (int j = 0; j < M.rows(); ++j) {
                        for (int k = 0; k < M.cols(); ++k) {
                            auto& v = M(j, k);
                            if (std::abs(v) < 1e-4) {
                                v = 0;
                            }
                        }
                    }
                    std::cout << M << std::endl;
                    std::cout << "Moment addition\n" << I << std::endl;
                    MyG += I;
                }
            }
            std::cout << "MyG\n" << MyG << std::endl;
            std::cout << "G for comparison\n" << G << std::endl;
        }
        if (false) {
            {
                std::cout << " Monomial evaluation (DOF operator)" << std::endl;
                std::cout << D << std::endl;
                mtao::MatXd R(c.local_sample_size(), c.monomial_size());
                R.setZero();
                auto EN = c.face_normals();
                for (auto&& [fidx_, sgn] : c.faces()) {
                    const auto& fidx = fidx_;
                    spdlog::warn("face info: {} {}", fidx, sgn);
                    int fsize = c.flux_size(fidx);
                    int offset = c.local_flux_index_offset(fidx);
                    auto samples = get_local_samples(fidx, 10000);
                    mtao::Vec3d N = mtao::eigen::stl2eigen(EN[fidx]);
                    tbb::parallel_for(int(0), int(fsize), [&](int j) {
                        tbb::parallel_for(
                            int(0), int(c.monomial_size()), [&](int col) {
                                auto mon = fmc.monomial_indexer().monomial(
                                    c.cell_index(), col);
                                double& r = R(offset + j, col) = 0;

                                for (int k = 0; k < samples.cols(); ++k) {
                                    auto s = samples.col(k);

                                    double a =
                                        fmc.flux_indexer().monomial_evaluation(
                                            fidx, j, s);
                                    double b = mon(s);
                                    r += a * b;
                                }

                                r /= samples.cols();
                            });
                    });

                    // std::cout << "MYBLOCK" << MyBlock << std::endl;
                }

                size_t mom_size = c.moment_size();

                if (mom_size > 0) {
                    {
                        size_t mom_off = c.local_moment_index_offset();
                        tbb::parallel_for(
                            int(0), int(c.moment_size()), [&](int j) {
                                int row = j + mom_off;
                                auto mr = c.monomial(j);
                                auto [mxexp_, myexp_, mzexp_] =
                                    vem::polynomials::three::index_to_exponents(
                                        j);
                                    const auto& mxexp = mxexp_;
                                    const auto& myexp = myexp_;
                                    const auto& mzexp = mzexp_;

                                tbb::parallel_for(
                                    int(0), int(c.monomial_size()),
                                    [&](int col) {
                                        auto mc = c.monomial(col);
                                        auto [Mxexp, Myexp, Mzexp] =
                                            vem::polynomials::three::
                                                index_to_exponents(col);

                                        int poly_index = vem::polynomials::
                                            three::exponents_to_index(
                                                mxexp + Mxexp, myexp + Myexp,
                                                mzexp + Mzexp);

                                        auto m = c.monomial(poly_index);
                                        R(row, col) = average_inside(
                                            [&](auto&& p) -> double {
                                                return mc(p);  // * mr(p);
                                            },
                                            // m,
                                            [&](auto&& p) -> bool {
                                                return c.is_inside(p);
                                            },
                                            bbox);
                                        /*
                                        R(row, col) = ival / vol;
                                        spdlog::info(
                                            "{} {} {} + {} {} {} =>
                                                {} /
                                                {} =
                                            {} ", mxexp, myexp, mzexp, Mxexp,
                                        Myexp, Mzexp, ival, vol, R(row, col));
                                        */
                                    });
                            });
                    }
                }
                std::cout << "Numerically integrated\n" << R << std::endl;
                std::cout << "Error:\n";
                std::cout << ((D - R).array().abs().array() < 1e-3)
                          << std::endl;
                std::cout << "Identity test" << std::endl;
            }

            // std::cout << "Identity\n" << clean(G.inverse() * B * R)
            // << std::endl;
        }

        std::cout << "Identity\n" << clean(G.inverse() * B * D) << std::endl;
        // std::cout << "Dirichlet grammian:\n";
        // std::cout << fmc.monomial_dirichlet_grammian() << std::endl;
        // std::cout << "regularized Dirichlet grammian:\n";
        // std::cout << fmc.regularized_monomial_dirichlet_grammian() <<
        // std::endl; std::cout << "Dirichlet sample grammian:\n";
        // std::cout
        // << fmc.sample_monomial_dirichlet_grammian() << std::endl;
        // auto SG = fmc.monomial_dirichlet_grammian();
        // auto Pi = fmc.dirichlet_projector();
        // auto Pi0 = fmc.l2_projector();
        //// std::cout << "Eval:\n";
        //// std::cout << clean(E) << std::endl;
        //// std::cout << "Dirichlet" << std::endl;
        //// std::cout << clean(Pi) << std::endl;
        //// std::cout << "L2" << std::endl;
        //// std::cout << clean(Pi0) << std::endl;

        // std::cout << "Dirichlet eval" << std::endl;
        // std::cout << clean(Pi * E) << std::endl;
        // std::cout << "L2 eval" << std::endl;
        // std::cout << clean(Pi0 * E) << std::endl;

        // std::cout << clean(Pi.transpose() * SG * Pi) << std::endl;
        break;
    }
    return;
}
int main(int argc, char* argv[]) {
    vem::VEMMesh3Creator creator;
    creator.mesh_filename = argv[1];

    creator.load_boundary_mesh();

    double vol = 0;
//#define DO_GRID
#if defined(DO_GRID)

    Eigen::AlignedBox<double, 3> bb;
    bb.min().setConstant(0);
    bb.max().setConstant(1);

    creator.grid_mesh_bbox = bb.cast<float>();
    creator.grid_mesh_dimensions[0] = 2;
    creator.grid_mesh_dimensions[1] = 2;
    creator.grid_mesh_dimensions[2] = 2;
    spdlog::info("Doing with mandoline");
    spdlog::info("Doing for grid");
    bool made_grid = creator.make_grid_mesh();
    {
        const auto& mesh = *creator.stored_mesh();

        for (int j = 0; j < mesh.cell_count(); ++j) {
            // spdlog::warn("Getting integrals for cell {}", j);
            auto v = monomial_cell_integrals(mesh, j, 3);
            vol += v(0);
            // std::cout << "===========" << std::endl;
            // std::cout << v.transpose() << std::endl;
            // std::cout << "===========" << std::endl;
            // std::cout << "===========" << std::endl;
            // std::cout << "===========" << std::endl;
            // std::cout << "===========" << std::endl;
        }
    }
    spdlog::info("bbox volume vs computed volume: {} vs {}", bb.sizes().prod(),
                 vol);
    {
        const auto& mesh = *creator.stored_mesh();
        test(mesh);
    }

#else
    auto bb =
        mtao::geometry::bounding_box(std::get<0>(*creator._held_boundary_mesh));
    bb.min().array() -= 5;
    bb.max().array() += 5;

    int N = 3;
    creator.grid_mesh_bbox = bb.cast<float>();
    creator.grid_mesh_dimensions[0] = N;
    creator.grid_mesh_dimensions[1] = N;
    creator.grid_mesh_dimensions[2] = N;
    spdlog::info("Doing with mandoline");
    bool made_mandoline = creator.make_mandoline_mesh();

    vol = 0;
    {
        const auto& mesh = *creator.stored_mesh();
        auto MCV = reinterpret_cast<const vem::MandolineVEMMesh3&>(mesh);
        MCV._ccm.write("/tmp/test.cutmesh");
        auto ccmV = MCV._ccm.cell_volumes();

        for (int j = 0; j < mesh.cell_count(); ++j) {
            auto v = monomial_cell_integrals(mesh, j, 3);
            // spdlog::warn(
            //    "Getting integrals for {} cell {}, got vol {} should
            //    be
            //    {}", MCV._ccm.is_cut_cell(j) ? "cut" : "grid", j,
            //    v(0), ccmV(j));
            if (!MCV._ccm.is_cut_cell(j)) {
                // spdlog::info("Valence of grid cell is : {}",
                //             mesh.cell_boundary_map.at(j).size());
                // for(auto&& [fidx,sgn]:
                //             mesh.cell_boundary_map.at(j)) {
                //    std::cout << fidx << ":" << (sgn?'-':'+') << " ";
                //}
                // std::cout << std::endl;
            }
            vol += v(0);
            // std::cout << "===========" << std::endl;
            // std::cout << v.transpose() << std::endl;
            // std::cout << "===========" << std::endl;
            // std::cout << "===========" << std::endl;
            // std::cout << "===========" << std::endl;
            // std::cout << "===========" << std::endl;
        }
    }

    spdlog::info("bbox volume vs computed volume: {} vs {}", bb.sizes().prod(),
                 vol);
    // spdlog::info("Made grid: {}; Made mandoline: {}", made_grid,
    //             made_mandoline);

    {
        const auto& mesh = *creator.stored_mesh();
        std::cout << mesh.cell_count() << std::endl;
        test(mesh);
    }
#endif
    return 0;
}
