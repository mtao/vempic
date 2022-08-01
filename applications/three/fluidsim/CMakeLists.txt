if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
        #src/sim_viewer.cpp
        #include/vem/fluidsim_3d/sim_viewer.hpp
        )
endif()
add_library(fluidsim_vem3
    src/fluidvem.cpp
    #src/fluidvem_function_projection.cpp
    #src/operator_cache.cpp
    #src/sim.cpp
    #src/sim_advection.cpp
    #src/sim_pressure.cpp
    #src/sim_scene.cpp
    #src/sim_buoyancy.cpp


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


