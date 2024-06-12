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

cmake_minimum_required(VERSION 3.8)

enable_testing()

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(sparrow-test CXX)
    find_package(sparrow REQUIRED CONFIG)
    set(SPARROW_INCLUDE_DIR ${sparrow_INCLUDE_DIRS})
endif ()

find_package(doctest REQUIRED)
#find_package(Threads)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

set(SPARROW_TESTS_SOURCES
    main.cpp
    array_data_creation.cpp
    test_algorithm.cpp
    test_allocator.cpp
    test_array_data_concepts.cpp
    test_array_data_creation.cpp
    test_array_data_factory.cpp
    test_array_data.cpp
    test_buffer_adaptor.cpp
    test_buffer.cpp
    test_c_data_interface.cpp
    test_dictionary_encoded_layout.cpp
    test_dynamic_bitset.cpp
    test_fixed_size_layout.cpp
    test_iterator.cpp
    test_memory.cpp
    test_mpl.cpp
    test_traits.cpp
    test_typed_array.cpp
    test_typed_array_timestamp.cpp
    test_variable_size_binary_layout.cpp
    better_junit_reporter.hpp
    junit.hpp
    junit.cpp
    junit_xml_writer.hpp
)
set(test_target "test_sparrow_lib")
add_executable(${test_target} ${SPARROW_TESTS_SOURCES})
target_link_libraries(${test_target} PRIVATE sparrow doctest::doctest)

include(doctest)
doctest_discover_tests(${test_target})

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(compiles_options
        /permissive-
        /WX # treat warnings as errors 
        /W4 # Baseline reasonable warnings
        /we4242 # 'identifier': conversion from 'type1' to 'type1', possible loss of data
        /we4244 # conversion from 'type1' to 'type_2', possible loss of data
        /we4254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data
        /we4263 # 'function': member function does not override any base class virtual member function
        /we4265 # 'classname': class has virtual functions, but destructor is not virtual instances of this class may not be destructed correctly
        /we4287 # 'operator': unsigned/negative constant mismatch
        /we4289 # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside the for-loop scope
        /we4296 # 'operator': expression is always 'boolean_value'
        /we4311 # 'variable': pointer truncation from 'type1' to 'type2'
        /we4545 # expression before comma evaluates to a function which is missing an argument list
        /we4546 # function call before comma missing argument list
        /we4547 # 'operator': operator before comma has no effect; expected operator with side-effect
        /we4549 # 'operator': operator before comma has no effect; did you intend 'operator'?
        /we4555 # expression has no effect; expected expression with side- effect
        /we4619 # pragma warning: there is no warning number 'number'
        /we4640 # Enable warning on thread un-safe static member initialization
        /we4826 # Conversion from 'type1' to 'type_2' is sign-extended. This may cause unexpected runtime behavior.
        /we4905 # wide string literal cast to 'LPSTR'
        /we4906 # string literal cast to 'LPWSTR'
        /we4928 # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
        /we5038 # data member 'member1' will be initialized after data member 'member2'
        /Zc:__cplusplus)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(compiles_options
        -pedantic # Warn on language extensions
        -Wall # reasonable and standard
        -Wcast-align # warn for potential performance problem casts
        -Wconversion # warn on type conversions that may lose data
        -Wdouble-promotion # warn if float is implicitly promoted to double
        -Werror # treat warnings as errors 
        -Wextra
        -Wformat=2 # warn on security issues around functions that format output (i.e., printf)
        -Wimplicit-fallthrough # Warns when case statements fall-through. (Included with -Wextra in GCC, not in clang)
        -Wmisleading-indentation #  warn if indentation implies blocks where blocks do not exist
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps catch hard to track down memory errors
        -Wnull-dereference # warn if a null dereference is detected
        -Wold-style-cast # warn for c-style casts
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -Wpedantic # warn if non-standard C++ is used
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wsign-conversion # warn on sign conversions
        -Wunused # warn on anything being unused
        $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches> # warn if if / else branches have duplicated code
        $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond> # warn if if / else chain has duplicated conditions
        $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op> # warn about logical operations being used where bitwise were probably wanted
        $<$<CXX_COMPILER_ID:GNU>:-Wno-subobject-linkage> # suppress warnings about subobject linkage
        $<$<CXX_COMPILER_ID:GNU>:-Wuseless-cast> # warn if you perform a cast to the same type
    ) 
endif()

target_compile_options(${test_target} PRIVATE ${compiles_options})

# We do not use non-standard C++
set_target_properties(${test_target} PROPERTIES CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(${test_target} PRIVATE cxx_std_20)

add_custom_target(run_tests
    COMMAND ${test_target}
    DEPENDS ${test_target}
    COMMENT "Running tests"
    USES_TERMINAL
)

set_target_properties(run_tests PROPERTIES FOLDER "Tests utilities")

set(JUNIT_REPORT_FILE ${CMAKE_CURRENT_BINARY_DIR}/test_sparrow_lib_report.xml)

add_custom_target(run_tests_with_junit_report
    COMMAND ${test_target} --reporters=better_junit --out=${JUNIT_REPORT_FILE} --no-path-filenames=true
    DEPENDS ${test_target}
    COMMENT "Running tests with JUnit report saved to: ${JUNIT_REPORT_FILE}"
    USES_TERMINAL
)

set_target_properties(run_tests_with_junit_report PROPERTIES FOLDER "Tests utilities")

