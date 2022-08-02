if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
        #src/sim_viewer.cpp
        #include/vem/fluidsim_3d/sim_viewer.hpp
        )
endif()
add_library(fluidsim_vem3
    src/fluidvem.cpp
    src/fluidvem_function_projection.cpp
    src/operator_cache.cpp
    src/sim.cpp
    src/sim_advection.cpp
    src/sim_pressure.cpp
    src/sim_scene.cpp
    src/sim_buoyancy.cpp


    ${VIEWER_SRCS}

    #src/fluidvem3_cell.cpp
    #include/vem/fluidsim_3d/sim.hpp
    #src/sim_serialization.cpp
    )
find_package(CHOLMOD REQUIRED)
message(STATUS "CHOLMOD INCLUDE: ${CHOLMOD_INCLUDES}")
target_link_libraries(fluidsim_vem3 PUBLIC vem std::filesystem
    #spqr ${CHOLMOD_LIBRARIES}
    nlohmann_json::nlohmann_json)
if(VEM_USE_PYTHON)
    TARGET_LINK_LIBRARIES(fluidsim_vem3 PUBLIC
    mtao::python)
endif()

target_include_directories(fluidsim_vem3 PUBLIC ${CHOLMOD_INCLUDES})
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(fluidsim_vem3 PUBLIC mtao::visualization
        )
ENDIF()

target_include_directories(fluidsim_vem3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_executable(fluidsim_3d src/fluidsim_3d.cpp)
target_link_libraries(fluidsim_3d vem fmt fluidsim_vem3 partio)

if(VEM_USE_OPENGL)

#ADD_EXECUTABLE(gradwavesim_gui_example2
#    gradwavesim_gui_example2.cpp
#    )
#TARGET_LINK_LIBRARIES(gradwavesim_gui_example2 gradwavesim_vem2 vem_visualize)

ADD_EXECUTABLE(fluidsim_gui_example3
    src/fluidsim_gui_example3.cpp
    )
TARGET_LINK_LIBRARIES(fluidsim_gui_example3 fluidsim_vem3 vem_visualize
    MagnumPlugins::PngImageConverter
    nlohmann_json::nlohmann_json
    )


endif()

ADD_EXECUTABLE(fluidsim3_projection_check
    src/fluidsim3_pressure_projection_check.cpp
    )
TARGET_LINK_LIBRARIES(fluidsim3_projection_check fluidsim_vem3 vem)

ADD_EXECUTABLE(fluidsim3_product_performance_test
    src/fluidsim3_product_performance_test.cpp
    )
TARGET_LINK_LIBRARIES(fluidsim3_product_performance_test fluidsim_vem3 vem)
