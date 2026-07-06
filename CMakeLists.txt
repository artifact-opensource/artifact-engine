cmake_minimum_required(VERSION 3.16)
project(artifact-engine VERSION 0.1.0 LANGUAGES C)

option(CPU_ONLY "Build CPU-only backend" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Shader compilation (GLSL → SPIR-V)
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
if(NOT GLSLC)
    message(WARNING "glslc not found — shaders must be pre-compiled")
endif()

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

# List of compute shaders
set(SHADERS
    matmul
    rmsnorm
    rope
    softmax
    silu
    add
    mul
    embedding
    dequant_q4k
)

# Compile each shader to SPIR-V
if(GLSLC)
    foreach(SHADER ${SHADERS})
        set(SHADER_SRC ${SHADER_DIR}/${SHADER}.comp)
        set(SHADER_SPV ${SPIRV_DIR}/${SHADER}.spv)
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${GLSLC} -O ${SHADER_SRC} -o ${SHADER_SPV}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling shader: ${SHADER}.comp → ${SHADER}.spv"
        )
        list(APPEND SPIRV_FILES ${SHADER_SPV})
    endforeach()
    
    add_custom_target(shaders ALL DEPENDS ${SPIRV_FILES})
endif()

# Source files
set(SOURCES
    src/main.c
    src/gguf.c
    src/cpu_compute.c
    src/engine.c
    src/http_server.c
    src/model_fetch.c
    src/frame_capture.c
    src/companion.c
    src/tokenizer.c
)

# Main executable
add_executable(artifact-engine ${SOURCES})

target_include_directories(artifact-engine PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

if(CPU_ONLY)
    target_compile_definitions(artifact-engine PRIVATE CPU_ONLY)
endif()

target_link_libraries(artifact-engine PRIVATE
    ${Vulkan_LIBRARIES}
    m
    pthread
)

if(GLSLC)
    add_dependencies(artifact-engine shaders)
endif()

# Install
install(TARGETS artifact-engine DESTINATION bin)
install(DIRECTORY ${SPIRV_DIR}/ DESTINATION share/artifact-engine/shaders)

# Cross-compilation helper
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(artifact-engine PRIVATE ws2_32)
    set_target_properties(artifact-engine PROPERTIES
        OUTPUT_NAME "artifact-engine"
        SUFFIX ".exe"
    )
endif()

# Print config
message(STATUS "Vulkan: ${Vulkan_LIBRARIES}")
message(STATUS "glslc: ${GLSLC}")
message(STATUS "Shaders: ${SHADERS}")
