# Copyright 2024 Man Group Operations Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.27)

# This is better specified per target, but cmake keeps ignoring these language version
# specification when building this project by itself, in particular the gnu extensions,
# so here we go.
# This will affects all following targets being defined.
set(CMAKE_CXX_EXTENSIONS OFF)

project(sparrow CXX)

include(CMakeDependentOption)

set(SPARROW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${SPARROW_INCLUDE_DIR}/sparrow/sparrow_version.hpp" sparrow_version_defines
     REGEX "constexpr int SPARROW_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${sparrow_version_defines})
    if(ver MATCHES "constexpr int SPARROW_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(SPARROW_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${SPARROW_VERSION_MAJOR}.${SPARROW_VERSION_MINOR}.${SPARROW_VERSION_PATCH})
message(STATUS "Building sparrow v${${PROJECT_NAME}_VERSION}")

# Build options
# =============

OPTION(BUILD_TESTS "Build sparrow test suite" OFF)
OPTION(BUILD_DOCS  "Build sparrow documentation" OFF)
OPTION(BUILD_EXAMPLES "Build sparrow examples" OFF)
OPTION(USE_DATE_POLYFILL "Use date polyfill implementation" ON)

include(CheckCXXSymbolExists)

if(cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  set(header version)
else()
  set(header ciso646)
endif()

check_cxx_symbol_exists(_LIBCPP_VERSION ${header} LIBCPP)
if(LIBCPP)
  message(STATUS "Using libc++")
  # Allow the use of not visible yet availabile features, such
  # as some formatter for new types.
  add_compile_definitions(_LIBCPP_DISABLE_AVAILABILITY)
endif()

# Linter options
# =============

OPTION(ACTIVATE_LINTER "Create targets to run clang-format" OFF)
cmake_dependent_option(ACTIVATE_LINTER_DURING_COMPILATION "Run linter during the compilation" ON "ACTIVATE_LINTER" OFF)

if(ACTIVATE_LINTER)
    include(cmake/clang-format.cmake)
    include(cmake/clang-tidy.cmake)
endif()

# Dependencies
# ============

set(SPARROW_INTERFACE_DEPENDENCIES "" CACHE STRING "List of dependencies to be linked to the sparrow target")

if (USE_DATE_POLYFILL)
    find_package(date CONFIG REQUIRED)
    list(APPEND SPARROW_INTERFACE_DEPENDENCIES date::date date::date-tz)
    add_compile_definitions(SPARROW_USE_DATE_POLYFILL)
endif()

# Build
# =====

set(SPARROW_HEADERS
    ${SPARROW_INCLUDE_DIR}/sparrow/algorithm.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/allocator.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/array.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/array_data.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/array_data_factory.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/buffer_adaptor.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/buffer.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/buffer_view.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/c_interface.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/config.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/contracts.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/data_traits.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/data_type.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/dynamic_bitset.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/fixed_size_layout.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/iterator.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/memory.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/mp_utils.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/null_layout.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/sparrow_version.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/typed_array.hpp
    ${SPARROW_INCLUDE_DIR}/sparrow/variable_size_binary_layout.hpp

    ${SPARROW_INCLUDE_DIR}/sparrow/details/3rdparty/float16_t.hpp
)

add_library(sparrow INTERFACE)

target_include_directories(sparrow INTERFACE
    $<BUILD_INTERFACE:${SPARROW_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(sparrow INTERFACE ${SPARROW_INTERFACE_DEPENDENCIES})

# We do not use non-standard C++
set_target_properties(sparrow PROPERTIES CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(sparrow INTERFACE cxx_std_20)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()


# Docs 
# ====
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Exacmples 
# ====
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS sparrow
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

   install(FILES ${SPARROW_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sparrow)

set(SPARROW_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for sparrowConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${SPARROW_CMAKECONFIG_INSTALL_DIR})

# sparrow is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from sparrowConfigVersion.cmake so that an sparrowConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_SPARROW_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_SPARROW_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${SPARROW_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${SPARROW_CMAKECONFIG_INSTALL_DIR})

