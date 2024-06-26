// Copyright 2024 Man Group Operations Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <string>

#include <sparrow/sparrow_version.hpp>

#include "better_junit_reporter.hpp"

TEST_CASE("version is readable")
{
    // TODO: once available on OSX, use `<format>` facility instead.
    // We only try to make sure the version valeus are printable, whatever their type.
    // AKA this is not written to be fancy but to force conversion to string.
    using namespace sparrow;
    [[maybe_unused]] const std::string printable_version = std::string("sparrow version : ")
                                                           + std::to_string(SPARROW_VERSION_MAJOR) + "."
                                                           + std::to_string(SPARROW_VERSION_MINOR) + "."
                                                           + std::to_string(SPARROW_VERSION_PATCH);
}
