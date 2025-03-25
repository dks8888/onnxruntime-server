cmake_minimum_required(VERSION 3.22.0)
project(onnxruntime_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

enable_testing()

set(ONNXRUNTIME_DIR "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime directory")
set(ONNXRUNTIME_USE_EXTENSIONS ON CACHE BOOL "Enable ONNX Runtime extensions")
set(ONNXRUNTIME_EXTENSIONS_PATH "/usr/local/lib/libortextensions.so" CACHE FILEPATH "Path to ONNX Runtime extensions library")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Подключаем кастомную библиотеку расширений явно
if(ONNXRUNTIME_USE_EXTENSIONS)
    message("Using ONNX Runtime Extensions from ${ONNXRUNTIME_EXTENSIONS_PATH}")
    add_definitions(-DONNXRUNTIME_USE_EXTENSIONS=1)
    link_libraries(${ONNXRUNTIME_EXTENSIONS_PATH})
endif()
