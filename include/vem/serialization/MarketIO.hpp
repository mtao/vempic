
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Desire NUENTSA WAKAM <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Modified from Eigen/unsupported/MarketIO.h
#include <Eigen/Core>
#include <fstream>

namespace market {

template <typename MatrixType>
bool loadMarketMatrix(MatrixType& mat, const std::string& filename) {
    typedef typename MatrixType::Scalar Scalar;
    std::ifstream in(filename.c_str(), std::ios::in);
    if (!in) return false;

    std::string line;
    int row(0), col(0);
    do {  // Skip comments
        std::getline(in, line);
        eigen_assert(in.good());
    } while (line[0] == '%');
    std::istringstream newline(line);
    newline >> row >> col;
    eigen_assert(row > 0 && col > 0);
    mat.resize(row, col);
    int i = 0;
    Scalar value;
    while (std::getline(in, line) && (i < row)) {
        auto r = mat.row(i);
        std::stringstream ss(line);
        for (int j = 0; j < col; ++j) {
            ss >> r(j);
        }
        ++i;
    }
    in.close();
    if (i != row) {
        std::cerr << "Unable to read all elements from file " << filename
                  << "\n";
        return false;
    }
    return true;
}

template <typename MatrixType>
bool saveMarketMatrix(const MatrixType& mat, const std::string& filename) {
    typedef typename MatrixType::Scalar Scalar;
    std::ofstream out(filename.c_str(), std::ios::out);
    if (!out) return false;

    out.flags(std::ios_base::scientific);
    out.precision(64);
    out << "%%MatrixMarket matrix array real general\n";
    out << mat.rows() << " " << mat.cols() << "\n";
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); ++j) {
            out << mat(i, j) << " ";
        }
        out << "\n";
    }
    out.close();
    return true;
}

}  // namespace market
