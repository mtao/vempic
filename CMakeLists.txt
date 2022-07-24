CMAKE_MINIMUM_REQUIRED(VERSION 3.10.1)
PROJECT(VEM)
SET(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(RANGES_CXX_STD 20)


#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
#if(COMPILER_SUPPORTS_MARCH_NATIVE)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#endif()

if(BUILD_TESTING AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

option(VEM_USE_OPENGL "use opengl" ON)
option(EMBREE_ISPC_SUPPORT "Embree ispc support" OFF)
option(VEM_USE_PYTHON "use python for some functions" ON)

option(LIBIGL_WITH_EMBREE "libigl use embree" ON)

set(CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )
find_package(nlohmann_json QUIET)
include(vem_fetch_dependencies)


find_package(nlohmann_json REQUIRED)

#find_package(igl)

find_package(Filesystem REQUIRED)


set(VEM_POLY_SRCS
    src/polynomials/gradient.cpp
    src/polynomials/partial_derivative.cpp
    src/polynomials/utils.cpp
    )

#add_library(vem 
set(STUFF
    src/edge_lengths.cpp
    src/cell_normals.cpp
    src/from_polygons.cpp
    src/from_mandoline.cpp
    src/from_mandoline3.cpp
    src/from_grid.cpp
    src/from_grid3.cpp
    src/mesh.cpp
    src/mesh3.cpp
    src/from_simplicial_matrices.cpp
    #src/monomial_basis_indexer.cpp
    src/monomial_basis_indexer_new.cpp
    src/monomial_edge_integrals.cpp 
    src/monomial_cell_integrals.cpp 
    src/monomial_cell_integrals3.cpp 
    src/monomial_face_integrals.cpp 
    src/monomial_face_integrals3.cpp 
    src/normals.cpp
    src/partitioned_coefficient_indexer.cpp
    src/polynomial_gradient.cpp
    src/polynomial_partial_derivative.cpp
    src/polynomial_utils.cpp
    src/rkhs_basis_indexer.cpp
    src/set_centroids_as_centers.cpp
    src/utils/boundary_facets.cpp
    src/utils/cells_adjacent_to_edge.cpp
    src/utils/cells_adjacent_to_face.cpp
    src/utils/cell_boundary_facets.cpp
    src/utils/face_boundary_facets.cpp
    src/utils/parent_maps.cpp
    src/utils/volumes.cpp
    src/utils/merge_cells.cpp
    src/utils/face_neighboring_cells.cpp
    src/utils/monomial_coefficient_projection.cpp
    src/utils/boundary_intersector.cpp
    src/utils/boundary_intersector3.cpp
    src/utils/in_triangle_mesh.cpp
    src/cell.cpp
    src/cell3.cpp
    src/creator2.cpp
    src/creator3.cpp
    src/point_moment_indexer.cpp
    src/point_moment_cell.cpp
    src/flux_moment_indexer.cpp
    src/flux_moment_cell.cpp
    src/flux_moment_indexer3.cpp
    src/flux_moment_cell3.cpp
    )

set(VEM_SRCS
    ${VEM_POLY_SRCS}
    )

add_library(vem ${VEM_SRCS})

target_link_libraries(vem PUBLIC mtao::geometry vem_serialization
    igl::embree
    #openvdb
    mandoline 
    )
target_include_directories(vem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(vem_serialization
    src/serialization/inventory.cpp
    src/serialization/serialize_eigen.cpp
    src/serialization/prioritizing_inventory.cpp
    src/serialization/frame_inventory.cpp
    src/serialization/serialize_text.cpp
    src/serialization/serialize_particles.cpp
    src/serialization/serialize_mesh.cpp
    #src/serialization/serialize_vdb.cpp
    )
target_link_libraries(vem_serialization PUBLIC mtao::geometry fmt
    nlohmann_json::nlohmann_json
    igl::core
    #openvdb Half
    PUBLIC std::filesystem cxxopts

    PRIVATE
    partio
    )
target_include_directories(vem_serialization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#
#
#if(VEM_USE_OPENGL)
#add_library(vem_visualize
#    src/visualize/vem_scalar_field_viewer.cpp
#    src/visualize/vem_mesh_creation_gui.cpp
#    src/visualize/inventory_viewer.cpp
#    )
#target_link_libraries(vem_visualize PUBLIC vem mtao::visualization
#    nlohmann_json::nlohmann_json)
#endif()
#
#
#if(BUILD_TESTING AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#    ADD_SUBDIRECTORY(tests)
#endif()

#add_subdirectory(poisson_2d)
#add_subdirectory(fluidsim_2d)
#add_subdirectory(fluidsim_3d)
#add_subdirectory(wavesim_2d)
##add_subdirectory(wavesim_3d)
###add_subdirectory(gradwavesim_2d)
#add_subdirectory(examples)
#add_subdirectory(viewer)
#add_subdirectory(tools)
