if(VEM_USE_OPENGL)
    SET(VIEWER_SRCS
    src/main.cpp
        )
add_executable(inventory_viewer_2
    ${VIEWER_SRCS})

TARGET_LINK_LIBRARIES(inventory_viewer_2 PUBLIC vem std::filesystem
    nlohmann_json::nlohmann_json mtao::visualization vem_visualize)
target_include_directories(inventory_viewer_2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

endif()



