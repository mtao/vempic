add_executable(load_triangle_mesh load_triangle_mesh.cpp)
target_link_libraries(load_triangle_mesh vem fmt poisson_vem2)

add_executable(load_grid_mesh load_grid_mesh.cpp)
target_link_libraries(load_grid_mesh vem fmt poisson_vem2)

add_executable(load_mandoline3 load_mandoline3.cpp)
target_link_libraries(load_mandoline3 vem fmt)

add_executable(make_collision_mesh3 make_collision_mesh3.cpp)
target_link_libraries(make_collision_mesh3 vem fmt)

add_executable(load_mandoline load_mandoline.cpp)
target_link_libraries(load_mandoline vem fmt)




add_executable(load_single_polygon load_single_polygon.cpp)
target_link_libraries(load_single_polygon vem fmt)
find_package(OpenMP REQUIRED)

ADD_EXECUTABLE(test_creating_cutmesh
    test_creating_cutmesh.cpp
    )
TARGET_LINK_LIBRARIES(test_creating_cutmesh vem)

ADD_EXECUTABLE(flux_moment_example flux_moment_example.cpp
    )
TARGET_LINK_LIBRARIES(flux_moment_example vem)

ADD_EXECUTABLE(flux_moment_example3 flux_moment_example3.cpp
    )
TARGET_LINK_LIBRARIES(flux_moment_example3 vem)

ADD_EXECUTABLE(moment_sample_reconstruction_experiment
    moment_sample_reconstruction_experiment.cpp
    )
TARGET_LINK_LIBRARIES(moment_sample_reconstruction_experiment poisson_vem2
    cxxopts
    OpenMP::OpenMP_CXX)

ADD_EXECUTABLE(moment_sample_reconstruction_experiment3
    moment_sample_reconstruction_experiment3.cpp
    )
TARGET_LINK_LIBRARIES(moment_sample_reconstruction_experiment3 vem
    cxxopts
    OpenMP::OpenMP_CXX)

ADD_EXECUTABLE(monomial_eval_test
    monomial_eval_test.cpp
    )
TARGET_LINK_LIBRARIES(monomial_eval_test poisson_vem2 )

