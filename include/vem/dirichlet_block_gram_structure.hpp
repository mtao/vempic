#include "vem/block_gram_structure.hpp"

namespace vem {
// per-cell dirichlet structure tool
struct DirichletBlockGramStructure : public BlockGramStructure {
    mtao::MatXd FG() const override;
    mtao::MatXd GF() const override;
    mtao::MatXd GG() const override;
    const VEMMesh2 &vem_mesh;
    const int index;
};
}// namespace vem
