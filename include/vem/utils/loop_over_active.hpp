#pragma once
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>
namespace vem::utils {

auto loop_over_active(const auto& container, const auto& active, auto&& f) {
    if (active.empty()) {
        for (auto&& c : container) {
            f(c);
        }
    } else {
        for (auto&& a : active) {
            f(container.at(a));
        }
    }
}

auto loop_over_active_indices(size_t size, const auto& active, auto&& f) {
    if (active.empty()) {
        for (size_t j = 0; j < size; ++j) {
            f(j);
        }
    } else {
        for (auto&& a : active) {
            f(a);
        }
    }
}
auto loop_over_active_tbb(const auto& container, const auto& active, auto&& f) {
    if (active.empty()) {
        tbb::parallel_for_each(container, f);
    } else {
        tbb::parallel_for_each(active, [&](size_t idx) {
            f(container.at(idx));
        });
    }
}

auto loop_over_active_indices_tbb(size_t size, const auto& active, auto&& f) {
    if (active.empty()) {
        tbb::parallel_for(size_t(0),size_t(size),f);
    } else {
        tbb::parallel_for_each(active, f);
    }
}

}  // namespace vem::utils
