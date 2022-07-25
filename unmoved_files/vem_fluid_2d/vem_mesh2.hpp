#pragma once
#include "vem_mesh.hpp"

class VEMMesh2 : public VEMMeshBase {
   public:
    mtao::ColVecs2i edges;
    mtao::ColVecs2d vertices;
    mtao::ColVecs2d centers;

    void initialize_interior_offsets(size_t edge_count);

    // these two are primarilyi for gui purposes
    void initialize_interior_offsets();
    size_t desired_edge_counts = 1;

    auto V(int index) const { return vertices.col(index); }
    auto C(int index) const { return centers.col(index); }
    auto E(int index) const { return edges.col(index); }
    size_t num_vertices() const override;
    size_t num_boundaries() const override;
    size_t num_edges() const;

    template <int D>
    mtao::RowVectors<double, D> polynomial_eval(
        size_t index, const mtao::Vec2d& p,
        const mtao::RowVectors<double, D>& coefficients) const;

    template <int D>
    mtao::RowVectors<double, D> polynomial_eval(
        size_t index, int sample_index,
        const mtao::RowVectors<double, D>& coefficients) const;

    mtao::RowVecXd polynomial_laplacian(
        size_t index, const mtao::Vec2d& p,
        const mtao::RowVecXd& coefficients) const;

    mtao::RowVecXd polynomial_laplacian(
        size_t index, int sample_index,
        const mtao::RowVecXd& coefficients) const;

    mtao::ColVecs2d polynomial_grad(size_t index, const mtao::Vec2d& p,
                                    const mtao::RowVecXd& coefficients) const;

    double winding_number(size_t cell_index, const mtao::Vec2d& p) const;
    bool is_inside(size_t cell_index, const mtao::Vec2d& p) const;

    mtao::ColVecs2d polynomial_grad(size_t index, int sample_index,
                                    const mtao::RowVecXd& coefficients) const;
    mtao::RowVecXd polynomial_entries(size_t index, const mtao::Vec2d& p) const;
    mtao::RowVecXd polynomial_entries(size_t index,
                                      size_t sample_index) const override;
    mtao::ColVecs2d polynomial_grad_entries(size_t index,
                                            const mtao::Vec2d& p) const;

    mtao::RowVecXd polynomial_laplacian_entries(size_t index,
                                                const mtao::Vec2d& p) const;

    static double polynomial_entry(size_t order, size_t index,
                                   const mtao::Vec2d& p);
    static mtao::Vec2d polynomial_grad_entry(size_t order, size_t index,
                                             const mtao::Vec2d& p);

    static double polynomial_laplacian_entry(size_t order, size_t index,
                                             const mtao::Vec2d& p);

    // this returns the edge indices in a line. similar to
    // boundary_Sample_indices_vec except this guarantees the order is along the
    // line of the edge in higher dimensions this would be duplicated by a mesh
    // topology
    std::vector<size_t> edge_samples(size_t boundary_index) const;

    // mtao::MatXd per_cell_projected_sample_gradients(size_t index) const;

    mtao::ColVecs2d cell_sample_positions(size_t cell_index) const;
    mtao::Vec2d edge_sample(size_t edge_index, size_t sample_index) const;
    mtao::Vec2d sample_position(size_t index) const;

    std::set<size_t> boundary_indices(size_t bound_index) const override;
    std::set<size_t> boundary_vertex_indices(size_t bound_index) const override;
    size_t edge_sample_index(size_t edge_index, size_t sample_index) const;
    // size_t sample_index(size_t edge_index, size_t sample_index) const
    // override;
    using VEMMeshBase::coefficient_size;
    size_t coefficient_size(size_t order) const override;

    // computes a laplacian by integrating monomials around the entire domain
    mtao::VecXd per_cell_per_monomial_integral(size_t index) const;
    mtao::VecXd per_cell_per_monomial_integral(size_t index,
                                               size_t max_order) const;
    mtao::VecXd monomial_integrals(
        const std::set<size_t>& disengaged_cells = {}) const;
    mtao::MatXd per_monomial_laplacian(size_t index) const;
    virtual Eigen::SparseMatrix<double> gradient_sample2poly() const {
        return {};
    }
    // mtao::MatXd per_monomial_divergence(size_t index) const;

    std::array<size_t, 2> monomial_index_to_powers(size_t index) const;
    size_t powers_to_monomial_index(size_t a, size_t b) const;
    // returns a 2*NCoeff x NCoeff matrix that represents the gradient
    Eigen::SparseMatrix<double> poly_gradient() const;
    Eigen::SparseMatrix<double> poly_d(int dim) const;
    Eigen::SparseMatrix<double> poly_dx() const;
    Eigen::SparseMatrix<double> poly_dy() const;

    // polynomial to polynomial divergence for each cell
    Eigen::SparseMatrix<double> divergence() const;
    Eigen::SparseMatrix<double> gradient() const;

    Eigen::SparseMatrix<double> laplacian(
        const std::set<size_t>& disengaged_cells = {}) const;

    virtual Eigen::SparseMatrix<double> regression_error_bilinear(
        const std::set<size_t>& disengaged_cells = {}) const = 0;
    virtual Eigen::SparseMatrix<double> integrated_divergence_poly2adj_sample(
        const std::set<size_t>& disengaged_cells = {}) const = 0;
    virtual mtao::MatXd laplacian_sample2sample(
        const std::set<size_t>& disengaged_cells = {}) const override;

    virtual Eigen::SparseMatrix<double> gradient_sample2sample() const {
        return {};
    }

    mtao::Mat2d boundary_basis(size_t boundary_index) const;

    // \nabla \cdot (x,y) = d_x x + d_y y
    mtao::VecXd per_cell_divergence(const mtao::VecXd& x,
                                    const mtao::VecXd& y) const;

    // polynomial to polynomial divergence for each cell
    mtao::VecXd divergence(const mtao::RowVecs2d& x) const;

    // poly to poly gradient for each cell
    mtao::RowVecs2d gradient(const mtao::VecXd& x) const;
    virtual Eigen::SparseMatrix<double> sample2cell_coefficients(
        const std::set<size_t>& disengaged_cells = {}) const = 0;

    Eigen::SparseMatrix<double> poly2sample(
        const std::set<int>& disengaged_cells = {}) const override;
};

template <int D>
mtao::RowVectors<double, D> VEMMesh2::polynomial_eval(
    size_t index, const mtao::Vec2d& p,
    const mtao::RowVectors<double, D>& coefficients) const {
    return polynomial_entries(index, p) * coefficients;
}

template <int D>
mtao::RowVectors<double, D> VEMMesh2::polynomial_eval(
    size_t index, int sample_index,
    const mtao::RowVectors<double, D>& coefficients) const {
    return polynomial_eval(index, sample_position(sample_index), coefficients);
}
