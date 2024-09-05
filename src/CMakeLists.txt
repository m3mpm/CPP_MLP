cmake_minimum_required(VERSION 3.5)

project(MLP VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_compile_options(-O3)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Charts)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Charts)

set(PROJECT_SOURCES
        resources/resource.qrc
        main.cpp
        controller/controller.cc
        controller/controller.h
        model/col_vector.cc
        model/col_vector.h
        model/data.h
        model/graph_layer.cc
        model/graph_layer.h
        model/graph_network.h
        model/layer.h
        model/matrix_layer.cc
        model/matrix_layer.h
        model/matrix_network.h
        model/matrix.cc
        model/matrix.h
        model/metrics.h
        model/network_service.cc
        model/network_service.h
        model/neural_network.cc
        model/neural_network.h
        model/neuron.cc
        model/neuron.h
        model/timer.h
        view/graphic.cpp
        view/graphic.h
        view/mainwindow.cpp
        view/mainwindow.h
        view/mainwindow.ui
        view/painter.h
        view/painter.cpp
        view/testing.h
        view/testing.cpp
        view/testing.ui
        view/validation.h
        view/validation.cpp
        view/validation.ui
        view/learning.h
        view/learning.cpp
        view/learning.ui
)

set (RESOURCE_PATH
        resources
)

set (RESOURCES
        ${RESOURCE_PATH}/resources.qrc
)


if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(MLP
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET MLP APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(MLP SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(MLP
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(MLP PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Charts)

set_target_properties(MLP PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(MLP)
endif()
