#include <fmt/format.h>

#include <chrono>
#include <vem/three/fluidsim/fluidvem.hpp>
#include <vem/three/fluidsim/sim.hpp>
#include <vem/three/from_grid.hpp>

using clock_type = std::chrono::high_resolution_clock;

auto make_mesh(int N) {
    Eigen::AlignedBox<double, 3> bb;
    bb.min().setConstant(0);
    bb.max().setConstant(1);
    auto mesh = vem::three::from_grid(bb, N, N, N);
    return mesh;
}

void mult_col(auto &&A, const auto &rhs, auto &res) {
    res.setZero(A.rows());

    int n = A.outerSize();

    for (int i = 0; i < n; ++i) {
        {
            for (typename std::decay_t<decltype(A)>::InnerIterator it(A, i); it;
                 ++it) {
                res.coeffRef(it.row()) += it.value() * rhs(it.col());
            }
        }
    }
}

void mult(auto &&A, const auto &rhs, auto &res) {
    res.resize(A.rows());

    int n = A.outerSize();

    for (int i = 0; i < n; ++i) {
        typename std::decay_t<decltype(res)>::Scalar tmp(0);
        {
            for (typename std::decay_t<decltype(A)>::InnerIterator it(A, i); it;
                 ++it)
                tmp += it.value() * rhs.coeff(it.index());
        }
        res(i) = tmp;
    }
}
void mult_omp(auto &&A, const auto &rhs, auto &res) {
    res.resize(A.rows());

    int n = A.outerSize();
    Eigen::initParallel();
    int threads = Eigen::nbThreads();

#pragma omp parallel for schedule( \
    dynamic, (n + threads * 4 - 1) / (threads * 4)) num_threads(threads)
    for (int i = 0; i < n; ++i) {
        typename std::decay_t<decltype(res)>::Scalar tmp(0);
        {
            for (typename std::decay_t<decltype(A)>::InnerIterator it(A, i); it;
                 ++it)
                tmp += it.value() * rhs.coeff(it.index());
        }
        res(i) = tmp;
    }
}

void mult_tbb(auto &&A, const auto &rhs, auto &res) {
    res.resize(A.rows());

    int n = A.outerSize();

    static tbb::affinity_partitioner ap;
    tbb::parallel_for(
        int(0), int(n),
        [&](int i) {
            typename std::decay_t<decltype(res)>::Scalar tmp(0);
            {
                for (typename std::decay_t<decltype(A)>::InnerIterator it(A, i);
                     it; ++it)
                    tmp += it.value() * rhs.coeff(it.index());
            }
            res(i) = tmp;
        },
        ap);
}

void test_projection_operators(int N, int D) {
    auto mesh = make_mesh(N);
    vem::three::fluidsim::FluidVEM3 fvem(mesh, D);

    spdlog::info("Building laplacian");
    Eigen::SparseMatrix<double> LC = fvem.sample_laplacian();
    Eigen::SparseMatrix<double, Eigen::RowMajor> L = LC;

    double flat_mult_time;
    double flat_col_mult_time;
    double builtin_mult_time;
    double tbb_mult_time;
    mtao::VecXd x_(L.rows());
    x_.setRandom();
    mtao::VecXd x = x_;
    int trials = 500;

    {
        spdlog::info("doing flat multiplication");
        auto start = clock_type::now();
        for (int j = 0; j < trials; ++j) {
            mult(L, x_, x);
        }
        auto end = clock_type::now();
        flat_mult_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count() /
            double(trials);
    }

    {
        spdlog::info("doing openmp multiplication");
        auto start = clock_type::now();
        for (int j = 0; j < trials; ++j) {
            mult_omp(L, x_, x);
        }
        auto end = clock_type::now();
        builtin_mult_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count() /
            double(trials);
    }

    {
        spdlog::info("doing flat col multiplication");
        auto start = clock_type::now();
        for (int j = 0; j < trials; ++j) {
            mult_col(LC, x_, x);
        }
        auto end = clock_type::now();
        flat_col_mult_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count() /
            double(trials);
    }

    {
        spdlog::info("doing tbb multiplication");
        auto start = clock_type::now();
        for (int j = 0; j < trials; ++j) {
            mult_tbb(L, x_, x);
        }
        auto end = clock_type::now();
        tbb_mult_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count() /
            double(trials);
    }
    fmt::print("{},{},{},{},{},{},{}\n", N, L.rows(), L.nonZeros(),
               flat_col_mult_time, flat_mult_time, builtin_mult_time,
               tbb_mult_time);
}

int main(int argc, char *argv[]) {  // test_projected_divergence(3);
    // test_projection_operators(2, 1);
    // test_projection_operators(2, 2);
    // test_projection_operators(2, 1);
    for (int j = 5; j < 50; ++j) {
        test_projection_operators(j, 2);
    }
}
