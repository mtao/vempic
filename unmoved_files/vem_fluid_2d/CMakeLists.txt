add_library(vem_fluid
    vem_mesh.hpp
    vem_mesh.cpp
    vem_mesh2.hpp
    vem_mesh2.cpp
    diwlevin_vem_mesh2.cpp
    pointwise_vem_mesh2.cpp
    interior_from_boundary.cpp
    cutmesh2_to_vemmesh.cpp
    sim.cpp
    )
TARGET_LINK_LIBRARIES(vem_fluid PUBLIC fluid_core)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vem_fluid PUBLIC OpenMP::OpenMP_CXX)
endif()
#/target_include_directories(vem_fluid PUBLIC /usr/include/mkl)
#/target_compile_definitions(vem_fluid PUBLIC -DEIGEN_USE_MKL_ALL)
#/    target_link_libraries(vem_fluid PUBLIC blas)

ADD_EXECUTABLE(vem_fluid_2d_viewer
    main_gl.cpp
    sim_vis.cpp
    )
TARGET_LINK_LIBRARIES(vem_fluid_2d_viewer vem_fluid)

ADD_EXECUTABLE(vem_fluid_2d_cli EXCLUDE_FROM_ALL
    main.cpp
    )
TARGET_LINK_LIBRARIES(vem_fluid_2d_cli vem_fluid)
ADD_EXECUTABLE(vem_poly_grad_test
    test_poly_grad.cpp
    )
TARGET_LINK_LIBRARIES(vem_poly_grad_test vem_fluid)
ADD_EXECUTABLE(vem_pressure_solve_check
    pressure_solve_check.cpp
    )
TARGET_LINK_LIBRARIES(vem_pressure_solve_check vem_fluid)
