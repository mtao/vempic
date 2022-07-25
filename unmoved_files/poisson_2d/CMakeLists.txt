if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
    src/constraint_viewer.cpp
        )
endif()
find_package(OpenMP REQUIRED)
add_library(poisson_vem2
    src/poisson_vem.cpp
    src/poisson_vem_cell.cpp
    src/constraints.cpp
    src/example_constraints.cpp
    ${VIEWER_SRCS}
    )

TARGET_LINK_LIBRARIES(poisson_vem2 PUBLIC vem std::filesystem
    OpenMP::OpenMP_CXX)
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(poisson_vem2 PUBLIC mtao::visualization mtao::python)
ENDIF()
target_include_directories(poisson_vem2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


