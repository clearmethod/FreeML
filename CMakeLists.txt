cmake_minimum_required(VERSION 3.16)

project(ML_2 LANGUAGES CXX)

include(CTest)

set(Example_OrGate_SRC Examples/OrGate/OrGate.cpp)
set(Example_OrGateBatched_SRC Examples/OrGate/OrGateBatched.cpp)
set(HelloFreeML_SRC Examples/HelloFreeML/HelloFreeML.cpp)
set(NanoLLM_SRC Examples/NanoLLM/NanoLLM.cpp)
set(DiffusionModel_SRC Examples/DiffusionModel/Diffusion.cpp)
set(ML2_TARGETS
      Example_OrGate
      Example_OrGateBatched
      HelloFreeML
      NanoLLM
      DiffusionModel
    )

foreach(tgt IN LISTS ML2_TARGETS)
  add_executable(${tgt} ${${tgt}_SRC})
  target_compile_features(${tgt} PRIVATE cxx_std_20)
  target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  if(MSVC)
    target_compile_options(${tgt} PRIVATE $<$<CONFIG:Debug>:/ZI>)
    target_link_options(${tgt} PRIVATE $<$<CONFIG:Debug>:/INCREMENTAL>)
  endif()

endforeach()

set(ML2_ASSET_TARGET_DIR "$<TARGET_FILE_DIR:ml2_main>")
add_custom_target(ml2_copy_assets ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/Examples/NanoLLM/Data"
          "Data"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/MatrixLibrary/GPU/DirectX11/Shaders"
          "GPU/DirectX11/Shaders"
)
add_dependencies(ml2_copy_assets ${ML2_TARGETS})

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
