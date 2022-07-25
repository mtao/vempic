if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
        )
endif()
add_library(wavesim_vem3
    src/sim.cpp
    src/sim_solver.cpp
    src/sim_scene.cpp
    src/wavevem3.cpp
    ${VIEWER_SRCS}
    )
TARGET_LINK_LIBRARIES(wavesim_vem3 PUBLIC vem std::filesystem
    mtao::python spqr cholmod
    nlohmann_json::nlohmann_json)
if(VEM_USE_OPENGL)
    TARGET_LINK_LIBRARIES(wavesim_vem3 PUBLIC mtao::visualization
        )
ENDIF()

target_include_directories(wavesim_vem3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)


