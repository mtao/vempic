#pragma once
#include <mtao/types.hpp>

namespace vem {
// This block Gram structure is designed to allow for the projection F into a
// subspace G

struct BlockGramStructure {
    // \int_E \langle \phi^F, \phi^F\rangle_H
    mtao::MatXd FF() const { return {}; }
    // \int_E \langle \phi^G, \phi^F\rangle_H
    mtao::MatXd FG() const = 0;
    // \int_E \langle \phi^F, \phi^G\rangle_H
    mtao::MatXd GF() const = 0;
    // \int_E \langle \phi^G, \phi^G\rangle_H
    mtao::MatXd GG() const = 0;

    mtao::MatXd schur_projector() const;
};

}// namespace vem
