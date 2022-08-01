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
    #src/monomial_basis_indexer.cpp
    src/utils/merge_cells.cpp
    #src/utils/monomial_coefficient_projection.cpp
    )

set(VEM_BASE_SRCS
    src/polygon_boundary_indices.cpp
    src/partitioned_coefficient_indexer.cpp
    src/utils/face_neighboring_cells.cpp
    src/utils/boundary_facets.cpp
)

set(VEM_TWO_SRCS
    src/two/mesh.cpp
    src/two/creator.cpp

    src/two/cell.cpp
    src/two/point_moment_cell.cpp
    src/two/flux_moment_cell.cpp

    src/two/point_sample_indexer.cpp
    src/two/flux_moment_indexer.cpp
    src/two/point_moment_indexer.cpp

    src/two/parent_maps.cpp
    src/two/set_centroids_as_centers.cpp
    src/two/normals.cpp
    src/two/edge_lengths.cpp
    src/two/cell_normals.cpp
    src/two/boundary_intersector.cpp
    src/two/boundary_facets.cpp
    src/two/cell_boundary_facets.cpp
    src/two/face_neighboring_cells.cpp
    src/two/cells_adjacent_to_edge.cpp

    src/two/volumes.cpp


    src/two/from_grid.cpp
    src/two/grid_mesh.cpp

    src/two/from_triangle_mesh.cpp
    src/two/triangle_mesh.cpp

    src/two/from_mandoline.cpp
    src/two/mandoline_mesh.cpp

    src/two/monomial_edge_integrals.cpp 
    src/two/monomial_cell_integrals.cpp 
    src/two/monomial_face_integrals.cpp 

    src/two/from_polygons.cpp
    src/two/polygon_mesh.cpp
    )

set(VEM_THREE_SRCS
    #src/from_mandoline3.cpp
    #src/from_grid3.cpp
    #src/three/mesh.cpp
    #src/three/flux_moment_indexer.cpp
    #src/monomial_cell_integrals3.cpp 
    #src/utils/cells_adjacent_to_face.cpp
    #src/monomial_face_integrals3.cpp 
    #src/three/volumes.cpp
    #src/three/cell_boundary_facets.cpp
    #src/three/cell.cpp
    #src/three/creator.cpp
    #src/three/flux_moment_cell3.cpp
    #src/utils/face_boundary_facets.cpp
    #src/three/boundary_intersector.cpp
    #src/three/in_triangle_mesh.cpp
    )

set(VEM_SRCS
    ${VEM_POLY_SRCS}
    ${VEM_BASE_SRCS}
    ${VEM_TWO_SRCS}
    #${VEM_THREE_SRCS}
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
if(VEM_USE_OPENGL)
add_library(vem_visualize
    src/two/visualize/vem_scalar_field_viewer.cpp
    src/two/visualize/vem_mesh_creation_gui.cpp
    src/two/visualize/inventory_viewer.cpp

    # Not used:
    #src/two/visualize/asset_viewer.cpp
    )
target_link_libraries(vem_visualize PUBLIC vem mtao::visualization
    nlohmann_json::nlohmann_json)
endif()
#
#
#if(BUILD_TESTING AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#    ADD_SUBDIRECTORY(tests)
#endif()

add_subdirectory(applications)
add_subdirectory(inventory_viewer_2)
#add_subdirectory(examples)
#add_subdirectory(viewer)
#add_subdirectory(tools)
