list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

set(TEST_SOURCES
  test_basis_builder.cpp
  test_polygon_clipper.cpp
  test_poly_integrator.cpp
  test_mapping_engine.cpp
  test_mapping_engine_3d.cpp
)

foreach(src ${TEST_SOURCES})
  get_filename_component(test_name ${src} NAME_WE)
  add_executable(${test_name} ${src})
  target_link_libraries(${test_name} PRIVATE l2map Catch2::Catch2WithMain)
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(${test_name} PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference_data"
  )
  catch_discover_tests(${test_name})
endforeach()
