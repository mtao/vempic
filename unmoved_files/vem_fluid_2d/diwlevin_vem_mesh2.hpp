#pragma once
#include "vem_mesh2.hpp"

class DIWLevinVEMMesh2 : public VEMMesh2 {
   public:
    bool integrated_edges = false;
    // diwlevin's A matrices
    // \int_a^b m(x) * n(x) for monomials m,n
    mtao::MatXd per_boundary_per_monomial_product(size_t index,
                                                  size_t boundary_index) const;
    mtao::MatXd per_cell_gramian(size_t cell_index) const;

    // diwlevin's B matrices
    // \int_a^b m(x) f(x) for data f(x)
    // the matrix is returned with columns associated with the sample indices of
    // the edge, in sorted order
    mtao::MatXd per_edge_per_monomial_linear_integral(size_t index,
                                                      size_t edge_index) const;
    mtao::MatXd per_cell_per_monomial_linear_integral(size_t index) const;
    // the diwlevin one

    // for a given cell
    mtao::MatXd per_edge_per_monomial_integrals(size_t index,
                                                size_t edge_index) const;
    // TODO: when i want to support more than one edge start filling this in
    // for a given cell
    // mtao::MatXd per_edge_per_cutedge_per_monomial_integrals(size_t index,
    //                                                size_t edge_index, size_t
    //                                                cutedge_index) const;

    std::vector<std::array<size_t, 3>> cell_cut_edges(size_t cell_index) const;

    // the best weights for reproducing the integrals of monomials
    mtao::VecXd per_cell_quadrature_weights(size_t cell_index) const;
    // the best weights for reproducing the integrals of monomials
    mtao::VecXd per_cell_edge_quadrature_weights(size_t cell_index) const;
    // returns a maps from coeffients to per-sample gradient values
    mtao::MatXd per_cell_gradients_poly2sample(size_t index) const;
    // returns a map from coefficients to per-sample gradient values
    mtao::MatXd per_cell_poly_gradients_sample2sample(size_t index) const;
    //  returns a map from coefficients to per-boundary boundary coefficients
    mtao::MatXd per_cell_integrated_projected_boundary_gradient(
        size_t index) const;
    Eigen::SparseMatrix<double> gradient_sample2sample() const override;

    virtual mtao::MatXd per_cell_hybrid_gradient(size_t index) const;
    mtao::MatXd per_cell_laplacian(size_t index) const override;
    mtao::MatXd poly_projection(size_t index) const;

    double per_edge_cut_edge_length(size_t edge_index) const;
    size_t per_edge_num_cut_edges(size_t edge_index) const;
    virtual std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
    orthogonal_neumann_entries(
        const std::map<size_t, double>& target_fluxes = {}) const override;

    virtual mtao::MatXd laplacian_sample2sample(
        const std::set<size_t>& disengaged_cells = {}) const override;
    Eigen::SparseMatrix<double> gradient_sample2poly() const override;
};
