pybind11_add_module(l2map_py bindings.cpp)
target_link_libraries(l2map_py PRIVATE l2map)
target_include_directories(l2map_py PRIVATE ${CMAKE_SOURCE_DIR}/include)
