find_program(CLANG_FORMAT clang-format)

if(NOT CLANG_FORMAT)
    message(WARNING "❗clang-format not found")
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

add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT} -i -style=file ${ALL_FILES_TO_FORMAT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on all files"
)

add_custom_target(
    clang-format_dry_run
    COMMAND ${CLANG_FORMAT} --dry-run -style=file ${ALL_FILES_TO_FORMAT}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running dry clang-format on all files"
)
