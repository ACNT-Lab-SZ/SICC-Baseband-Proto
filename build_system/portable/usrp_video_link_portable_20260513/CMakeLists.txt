cmake_minimum_required(VERSION 3.20)

project(usrp_ldpc_ofdm_link LANGUAGES C CXX)

option(ENABLE_GPU_FULL_PIPELINE "Build the experimental CUDA-resident OFDM/FEC pipeline module" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED UHD_ROOT)
    if(DEFINED ENV{UHD_ROOT})
        set(UHD_ROOT "$ENV{UHD_ROOT}" CACHE PATH "UHD install root")
    elseif(DEFINED ENV{UHD_PKG_PATH})
        set(UHD_ROOT "$ENV{UHD_PKG_PATH}" CACHE PATH "UHD install root")
    else()
        set(UHD_ROOT "" CACHE PATH "UHD install root")
    endif()
endif()

find_path(UHD_INCLUDE_DIR
    NAMES uhd/usrp/multi_usrp.hpp
    HINTS "${UHD_ROOT}/include"
)

find_library(UHD_LIBRARY
    NAMES uhd
    HINTS "${UHD_ROOT}/lib"
)

if(NOT UHD_INCLUDE_DIR OR NOT UHD_LIBRARY)
    message(FATAL_ERROR "Could not find UHD. Set -DUHD_ROOT=<uhd-install-root>.")
endif()

add_executable(uhd_ldpc_ofdm_link
    src/fec_config.cpp
    src/matrix_loader.cpp
    src/rx_decoder_router.cpp
    src/uhd_c_radio.c
    src/uhd_radio_session.cpp
    src/uhd_ldpc_ofdm_link.cpp
)

add_executable(fec_offline_validation
    src/fec_config.cpp
    src/matrix_loader.cpp
    src/rx_decoder_router.cpp
    tests/fec_offline_validation.cpp
)

target_include_directories(uhd_ldpc_ofdm_link PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${UHD_INCLUDE_DIR}"
)

target_include_directories(fec_offline_validation PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(uhd_ldpc_ofdm_link PRIVATE
    "${UHD_LIBRARY}"
)

if(WIN32)
    target_link_libraries(uhd_ldpc_ofdm_link PRIVATE ws2_32)
endif()

if(ENABLE_GPU_FULL_PIPELINE)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gpu_ofdm_pipeline STATIC
        src/gpu_ofdm_pipeline.cu
    )
    target_compile_features(gpu_ofdm_pipeline PUBLIC cxx_std_17)
    set_target_properties(gpu_ofdm_pipeline PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 89
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(gpu_ofdm_pipeline PUBLIC CUDA::cudart CUDA::cufft)
    if(MSVC)
        target_compile_options(gpu_ofdm_pipeline PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    else()
        target_compile_options(gpu_ofdm_pipeline PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-Wall>
        )
    endif()
    add_executable(gpu_ofdm_pipeline_bench
        tests/gpu_ofdm_pipeline_bench.cpp
    )
    target_include_directories(gpu_ofdm_pipeline_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(gpu_ofdm_pipeline_bench PRIVATE gpu_ofdm_pipeline)
    if(MSVC)
        target_compile_options(gpu_ofdm_pipeline_bench PRIVATE /O2 /EHsc /permissive-)
        target_compile_definitions(gpu_ofdm_pipeline_bench PRIVATE NOMINMAX)
    else()
        target_compile_options(gpu_ofdm_pipeline_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
    endif()
    target_compile_definitions(uhd_ldpc_ofdm_link PRIVATE HAVE_GPU_FULL_PIPELINE=1)
    target_link_libraries(uhd_ldpc_ofdm_link PRIVATE gpu_ofdm_pipeline)
endif()

set(CUDA_OSD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda_osd" CACHE PATH "CUDA BP-OSD project root")
set(CUDA_OSD_IMPORT_LIB "${CUDA_OSD_ROOT}/build/vs2022/Release/cuda_osd_matlab_ccsds_sync3.lib")
set(CUDA_OSD_RUNTIME_DLL "${CUDA_OSD_ROOT}/build/vs2022/Release/cuda_osd_matlab_ccsds_sync3.dll")
if(EXISTS "${CUDA_OSD_ROOT}/include/matlab_decoder_c_api.h" AND
   EXISTS "${CUDA_OSD_IMPORT_LIB}" AND
   EXISTS "${CUDA_OSD_RUNTIME_DLL}")
    target_compile_definitions(uhd_ldpc_ofdm_link PRIVATE HAVE_CUDA_OSD=1)
    target_include_directories(uhd_ldpc_ofdm_link PRIVATE "${CUDA_OSD_ROOT}/include")
    target_link_libraries(uhd_ldpc_ofdm_link PRIVATE "${CUDA_OSD_IMPORT_LIB}")
    add_custom_command(TARGET uhd_ldpc_ofdm_link POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CUDA_OSD_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:uhd_ldpc_ofdm_link>/cuda_osd_matlab_ccsds_sync3.dll"
        VERBATIM
    )

    add_executable(gpu_bp_osd_awgn_bench
        tests/gpu_bp_osd_awgn_bench.cpp
    )
    target_compile_definitions(gpu_bp_osd_awgn_bench PRIVATE HAVE_CUDA_OSD=1)
    target_include_directories(gpu_bp_osd_awgn_bench PRIVATE "${CUDA_OSD_ROOT}/include")
    target_link_libraries(gpu_bp_osd_awgn_bench PRIVATE "${CUDA_OSD_IMPORT_LIB}")
    add_custom_command(TARGET gpu_bp_osd_awgn_bench POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CUDA_OSD_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:gpu_bp_osd_awgn_bench>/cuda_osd_matlab_ccsds_sync3.dll"
        VERBATIM
    )
else()
    message(WARNING "CUDA BP-OSD library not found under CUDA_OSD_ROOT=${CUDA_OSD_ROOT}; --decoder cuda-bp-osd will be unavailable.")
endif()

set(UHD_RUNTIME_DLL "${UHD_ROOT}/bin/uhd.dll")
if(EXISTS "${UHD_RUNTIME_DLL}")
    add_custom_command(TARGET uhd_ldpc_ofdm_link POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${UHD_RUNTIME_DLL}"
            "$<TARGET_FILE_DIR:uhd_ldpc_ofdm_link>/uhd.dll"
        VERBATIM
    )
endif()

set(UHD_LIBUSB_DLL "${UHD_ROOT}/bin/libusb-1.0.dll")
if(EXISTS "${UHD_LIBUSB_DLL}")
    add_custom_command(TARGET uhd_ldpc_ofdm_link POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${UHD_LIBUSB_DLL}"
            "$<TARGET_FILE_DIR:uhd_ldpc_ofdm_link>/libusb-1.0.dll"
        VERBATIM
    )
endif()

if(MSVC)
    target_compile_options(uhd_ldpc_ofdm_link PRIVATE /O2 /EHsc /permissive-)
    target_compile_options(fec_offline_validation PRIVATE /O2 /EHsc /permissive-)
    target_compile_definitions(uhd_ldpc_ofdm_link PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_definitions(fec_offline_validation PRIVATE NOMINMAX)
    if(TARGET gpu_bp_osd_awgn_bench)
        target_compile_options(gpu_bp_osd_awgn_bench PRIVATE /O2 /EHsc /permissive-)
        target_compile_definitions(gpu_bp_osd_awgn_bench PRIVATE NOMINMAX)
    endif()
    if(TARGET gpu_ofdm_pipeline_bench)
        target_compile_options(gpu_ofdm_pipeline_bench PRIVATE /O2 /EHsc /permissive-)
        target_compile_definitions(gpu_ofdm_pipeline_bench PRIVATE NOMINMAX)
    endif()
else()
    target_compile_options(uhd_ldpc_ofdm_link PRIVATE -O3 -Wall -Wextra -Wpedantic)
    target_compile_options(fec_offline_validation PRIVATE -O3 -Wall -Wextra -Wpedantic)
    if(TARGET gpu_bp_osd_awgn_bench)
        target_compile_options(gpu_bp_osd_awgn_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
    endif()
    if(TARGET gpu_ofdm_pipeline_bench)
        target_compile_options(gpu_ofdm_pipeline_bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
    endif()
endif()
