#pragma once
#include <Eigen/Sparse>
#include <map>
#include <mtao/types.hpp>
#include <set>
#include <vector>

class VEMMeshBase {
   public:
    // using SeqType = Eigen::ArithmeticSequence
    VEMMeshBase();
    ~VEMMeshBase();

    // returns a sparse matrix from the left index space to the right index
    // space. when to is empty it's assumed that to is the global index space.
    // opt arg size is used if to is empty
    static Eigen::SparseMatrix<double> index_map(
        const std::vector<size_t>& from, const std::vector<size_t>& to = {},
        size_t size = 0);
    Eigen::SparseMatrix<double> local_to_world(
        const std::vector<size_t>& from) const;
    Eigen::SparseMatrix<double> cell_to_world(size_t index) const;
    Eigen::SparseMatrix<double> boundary_to_world(size_t index) const;

    Eigen::SparseMatrix<double> boundary_facets_to_world(
        size_t cell_index) const;

    mtao::MatXd poly_coefficient_matrix(
        size_t index, const std::vector<size_t>& indices) const;
    mtao::MatXd poly_coefficient_matrix(size_t index) const;

    virtual mtao::RowVecXd polynomial_entries(size_t cell_index,
                                              size_t sample_index) const {
        return {};
    }

    virtual mtao::MatXd per_cell_laplacian(size_t index) const { return {}; }

    virtual size_t num_vertices() const { return {}; }
    size_t num_samples() const;
    size_t per_cell_num_samples(size_t cell_index) const;
    virtual size_t num_boundaries() const { return {}; }
    size_t num_boundary_samples() const;
    size_t num_cells() const;
    virtual size_t coefficient_size(size_t order) const { return {}; }

    // number of polynomial coefficients per polynomial
    size_t coefficient_size() const;

    // num cells * coefficient size; useful for data living in per-cell
    // polynomial space
    size_t polynomial_size() const;

    // these offsets are not aware of vertices, i.e index[0] is associated with
    // the first vertex
    std::array<size_t, 2> boundary_internal_index_range(
        size_t boundary_index) const;
    size_t boundary_internal_index_offset(size_t boundary_index) const;
    size_t boundary_internal_sample_index(size_t bound_index,
                                          size_t sample_index) const;

    // helpers for operating on all samples in a cell at once
    // size_t num_samples_on_boundary(size_t boundary_index) const;
    size_t num_interior_samples_on_boundary(size_t boundary_index) const;

    // the boundary facets indexed for a cell
    virtual std::set<size_t> boundary_indices(size_t cell_index) const {
        return {};
    }
    std::vector<size_t> boundary_indices_vec(size_t cell_index) const;
    // just returns the vertices on the boundary, no samples!
    virtual std::set<size_t> boundary_vertex_indices(size_t bound_index) const {
        return {};
    }
    std::vector<size_t> boundary_vertex_indices_vec(size_t bound_index) const;
    std::set<size_t> cell_sample_indices(size_t cell_index) const;
    std::vector<size_t> cell_sample_indices_vec(size_t cell_index) const;

    // these two return t he number of samples, including the input vertices
    std::set<size_t> boundary_sample_indices(size_t bound_index) const;
    std::vector<size_t> boundary_sample_indices_vec(size_t bound_index) const;
    Eigen::SparseMatrix<double> cell_sample_to_world_sample(
        size_t cell_index) const;

    virtual Eigen::SparseMatrix<double> poly2sample(
        const std::set<int>& disengaged_cells = {}) const {
        return {};
    }
    // helpers for individual samples
    // virtual size_t sample_index(size_t bound_index, size_t sample_index)
    // const { return {}; }

    // given the index of a sample, return the boundary facet that it comes from
    // returns {} if no value is found (index > num samples or is vertex)
    std::optional<size_t> get_sample_parent_boundary(size_t sample_index) const;

    std::map<size_t, std::array<int, 2>> coboundary() const;

    virtual mtao::MatXd laplacian_sample2sample(
        const std::set<size_t>& disengaged_cells) const;
    // the orthogonal neumann samples can be determined by a derived class
    mtao::VecXd laplace_problem(
        const std::map<size_t, double>& constrained_vertices,
        const std::map<size_t, double>& orthogonal_neumann_samples = {},
        const std::set<size_t>& disengaged_cells = {}) const;
    mtao::VecXd poisson_problem(
        const mtao::VecXd& coefficient_rhs,
        const std::map<size_t, double>& constrained_vertices = {},
        const std::map<size_t, double>& orthogonal_neumann_samples = {},
        const std::set<size_t>& disengaged_cells = {}) const;
    // returns a matrix and vector representing the dirichlet and cauchy
    // boundary conditions
    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> dirichlet_entries(
        const std::map<size_t, double>& dirichlet_vertices) const;

    // the inputs are desired values of N \cdot \nabla f
    // it's up to the derived class to determine how the scalar is applied - i.e
    // it may be easy to assume that this is distributed across the boundary
    virtual std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
    orthogonal_neumann_entries(
        const std::map<size_t, double>& target_fluxes = {}) const {
        return {};
    }

    // std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> cauchy_entries(
    //    const std::map<size_t, double>& dirichlet_vertices
    //    const std::map<size_t, double>& neumann_edges
    //    ) const;

    std::vector<std::map<size_t, bool>> cells;
    std::vector<size_t> boundary_sample_offsets;
    size_t order = 2;
};

