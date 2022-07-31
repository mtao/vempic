if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
        src/constraint_viewer.cpp
        )
endif()
find_package(OpenMP REQUIRED)
add_library(poisson_vem2
    src/poisson.cpp
    src/cell.cpp
    src/constraints.cpp
    src/example_constraints.cpp
    ${VIEWER_SRCS}
    )

TARGET_LINK_LIBRARIES(poisson_vem2 PUBLIC vem std::filesystem
    OpenMP::OpenMP_CXX)
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(poisson_vem2 PUBLIC mtao::visualization mtao::python vem_visualize)
ENDIF()
target_include_directories(poisson_vem2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)



ADD_EXECUTABLE(poisson2_example
    src/poisson2_example.cpp
    )
TARGET_LINK_LIBRARIES(poisson2_example poisson_vem2)
ADD_EXECUTABLE(poisson_solution_viewer2
    src/poisson_solution_viewer2.cpp
    )
if(VEM_USE_OPENGL)
TARGET_LINK_LIBRARIES(poisson_solution_viewer2 poisson_vem2 vem_visualize

    MagnumPlugins::PngImageConverter
    )
endif()
