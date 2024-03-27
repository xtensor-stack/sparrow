find_program(CLANG_FORMAT clang-format)

if(NOT CLANG_FORMAT)
    message(WARNING "clang-format not found")
    return()
endif()

# list all files to format

set(
    FORMAT_PATTERNS
    include/*.hpp
    test/*.cpp
    test/*.hpp
    CACHE STRING
    "; separated patterns relative to the project source dir to format"
)

set(ALL_FILES_TO_FORMAT "")
foreach(PATTERN ${FORMAT_PATTERNS})
    file(GLOB_RECURSE FILES_TO_FORMAT ${CMAKE_SOURCE_DIR}/${PATTERN})
    list(APPEND ALL_FILES_TO_FORMAT ${FILES_TO_FORMAT})
endforeach()

string(REPLACE ";" "\n" FILES_TO_FORMAT "${ALL_FILES_TO_FORMAT}")

# generate a txt file with one file path per line
message(STATUS "Generating clang-format input file list")
set(CLANG_FORMAT_INPUT_FILES ${CMAKE_BINARY_DIR}/clang-format_input_files.txt)
file(WRITE ${CLANG_FORMAT_INPUT_FILES} ${FILES_TO_FORMAT})

add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT} -i -style=file --files=${CLANG_FORMAT_INPUT_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on all files"
)

add_custom_target(
    clang-format_dry_run
    COMMAND ${CLANG_FORMAT} --dry-run -i -style=file --files=${CLANG_FORMAT_INPUT_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running dry clang-format on all files"
)
