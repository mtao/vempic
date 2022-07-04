
#include "vem/serialization/serialize_eigen.hpp"
#include <spdlog/spdlog.h>

#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <spdlog/spdlog.h>
#include "vem/serialization/MarketIO.hpp"

#include <eigen3/unsupported/Eigen/SparseExtra>
/*
#include <mtao/python/load_python_function.hpp>

namespace {
    using namespace mtao::python;



}
*/

namespace vem::serialization {
void serialize_sparse_matrix(Inventory& inventory, const std::string& name,
                             const Eigen::SparseMatrix<double>& A) {
    auto p = inventory.get_new_asset_path(name);
    bool s = Eigen::saveMarket(A, p);
    inventory.asset_metadata(name)["storage_type"] = "eigen_market";
    if (!s) {
        spdlog::warn("Was unable to serialize a sparse matrix");
    }
    inventory.asset_metadata(name)["data_type"] = "SparseMatrix<double>";
}

template <typename T>
void serialize_VecX(Inventory& inventory, const std::string& name,
                    const mtao::VectorX<T>& A) {
    auto p = inventory.get_new_asset_path(name);

    bool s = Eigen::saveMarketVector(A, p);
    inventory.asset_metadata(name)["storage_type"] = "eigen_market";
    if (!s) {
        spdlog::warn("Was unable to serialize a vecxd");
    }
}
void serialize_VecXd(Inventory& inventory, const std::string& name,
                     const mtao::VecXd& A) {
    serialize_VecX<double>(inventory, name, A);
    inventory.asset_metadata(name)["data_type"] = "VecXd";
}

void serialize_VecXi(Inventory& inventory, const std::string& name,
                     const mtao::VecXi& A) {
    serialize_VecX<int>(inventory, name, A);
    inventory.asset_metadata(name)["data_type"] = "VecXi";
}
void serialize_obj(Inventory& inventory, const std::string& name,
                   const mtao::ColVecs3d& V, const mtao::ColVecs3i& F) {
    auto p = inventory.get_new_asset_path(name);
    igl::writeOBJ(p, V.transpose(), F.transpose());
    inventory.asset_metadata(name)["storage_type"] = "obj";
}

template <int D>
void serialize_points(Inventory& inventory, const std::string& name,
                      const mtao::ColVectors<double, D>& P) {

    auto p = inventory.get_new_asset_path(name);

    // NOTE that i want more newlines to make it easier to read rather than 2-3 long rows
    // therefore there is a transpose

    bool s = market::saveMarketMatrix(P.transpose(), p);
    inventory.asset_metadata(name)["storage_type"] = "market";
    if (!s) {
        spdlog::warn("Was unable to serialize points");
    }


    //Inventory subinv = inventory.make_subinventory(name);
    //subinv.add_metadata("storage_type", "point_collection");
    //subinv.add_metadata("dimension", D);
    //subinv.add_metadata("count", P.cols());
    //for (int j = 0; j < D; ++j) {
    //    std::string name(1, 'a' + j);
    //    serialize_VecXd(subinv, name, P.row(j).transpose());
    //}

    //// dont touch the file because the subinv will be made by default
    //auto p = inventory.get_new_asset_path(name, {}, /*directory=*/true,
    //                                      /*touch_file=*/false, false);
    //inventory.asset_metadata(name)["storage_type"] = "inventory";
    //inventory.asset_metadata(name)["data_type"] = fmt::format("colvecs{}d", D);
}
void serialize_points2(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs2d& P) {
    serialize_points(inventory, name, P);
}
void serialize_points3(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs3d& P) {
    serialize_points(inventory, name, P);
}
void serialize_points4(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs4d& P) {
    serialize_points(inventory, name, P);
}
void serialize_points5(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs5d& P) {
    serialize_points(inventory, name, P);
}

void serialize_points6(Inventory& inventory, const std::string& name,
                       const mtao::ColVecs6d& P) {
    serialize_points(inventory, name, P);
}

Eigen::SparseMatrix<double> deserialize_sparse_matrix(
    const Inventory& inventory, const std::string& name) {
    std::string type = inventory.asset_metadata(name).at("storage_type");
    if (type == "eigen_market") {
        Eigen::SparseMatrix<double> A;
        Eigen::loadMarket(A, inventory.get_asset_path(name));
        return A;
    } else {
        spdlog::error("Expected a market_type but got a type {}", type);
        return {};
    }
}
template <typename T>
mtao::VectorX<T> deserialize_VecX(const Inventory& inventory,
                                  const std::string& name) {
    std::string type = inventory.asset_metadata(name).at("storage_type");
    if (type == "eigen_market") {
        mtao::VectorX<T> A;
        Eigen::loadMarketVector(A, inventory.get_asset_path(name));
        return A;
    } else {
        spdlog::error("Expected a market_type but got a type {}", type);
        return {};
    }
}
mtao::VecXd deserialize_VecXd(const Inventory& inventory,
                              const std::string& name) {
    return deserialize_VecX<double>(inventory, name);
}

mtao::VecXi deserialize_VecXi(const Inventory& inventory,
                              const std::string& name) {
    return deserialize_VecX<int>(inventory, name);
}

std::tuple<mtao::ColVecs3d, mtao::ColVecs3i> deserialize_obj3(
    const Inventory& inventory, const std::string& name) {
    std::string type = inventory.asset_metadata(name).at("storage_type");
    auto p = inventory.get_asset_path(name);
    if (type == "obj") {
        Eigen::MatrixXd VV;
        Eigen::MatrixXi FF;
        igl::read_triangle_mesh(p, VV, FF);
        return {VV.transpose(), FF.transpose()};

    } else {
        spdlog::error("Expected a obj but got a type {}", type);
        return {};
    }
}
template <int D>
mtao::ColVectors<double, D> deserialize_points(const Inventory& inventory,
                                               const std::string& name) {
    std::string type = inventory.asset_metadata(name).at("storage_type");
    if(type == "market") {
        mtao::ColVectors<double, D> R;
        Eigen::MatrixXd RR;
        market::loadMarketMatrix(RR, inventory.get_asset_path(name));
        //std::cout << "SR:\n" << RR << std::endl;
        // this was a trick to store each point as a row instead of a col
        R = RR.transpose();
        //spdlog::info("Unloaded a market of size {} {}", RR.rows(), RR.cols());
        return R;

    } else if (type == "inventory") {
        const Inventory subinv = inventory.get_subinventory(name);
        mtao::ColVectors<double, D> R(D,
                                      subinv.metadata().at("count").get<int>());
        for (int j = 0; j < R.rows(); ++j) {
            std::string name(1, 'a' + j);
            R.row(j) = deserialize_VecXd(subinv, name).transpose();
        }

        return R;
    } else {
        spdlog::error("Expected a inventory but got a type {}", type);
        return {};
    }
}

mtao::ColVecs2d deserialize_points2(const Inventory& inventory,
                                    const std::string& name) {
    return deserialize_points<2>(inventory, name);
}
mtao::ColVecs3d deserialize_points3(const Inventory& inventory,
                                    const std::string& name) {
    return deserialize_points<3>(inventory, name);
}
mtao::ColVecs4d deserialize_points4(const Inventory& inventory,
                                    const std::string& name) {
    return deserialize_points<4>(inventory, name);
}
mtao::ColVecs5d deserialize_points5(const Inventory& inventory,
                                    const std::string& name) {
    return deserialize_points<5>(inventory, name);
}
mtao::ColVecs6d deserialize_points6(const Inventory& inventory,
                                    const std::string& name) {
    return deserialize_points<6>(inventory, name);
}
}  // namespace vem::serialization
