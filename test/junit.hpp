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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace sparrow
{
    struct JUnitProperties
    {
        std::string m_name;
        std::vector<std::string> m_value;
    };

    struct JUnitTestMessage
    {
        JUnitTestMessage(std::string message, std::string type, std::string details, size_t line);
        JUnitTestMessage(std::string message, std::string details, size_t line);

        std::string m_message;
        std::string m_type;
        std::string m_details;
        size_t m_line;
    };

    struct JUnitTestCase
    {
        JUnitTestCase(
            std::string classname,
            std::string name,
            double time,
            std::string file,
            size_t line,
            std::vector<JUnitTestMessage> failures,
            std::vector<JUnitTestMessage> errors,
            size_t assertions_count,
            bool skipped = false
        );

        std::string m_classname;
        std::string m_name;
        double m_time = 0;
        std::string m_file;
        size_t m_line = 0;
        std::vector<JUnitTestMessage> m_failures;
        std::vector<JUnitTestMessage> m_errors;
        size_t m_assertions_count = 0;
        bool m_skipped = false;
    };

    struct JUnitTestSuite
    {
        JUnitTestSuite(std::string name, std::string timestamp)
            : m_name(std::move(name))
            , m_timestamp(std::move(timestamp))
        {
        }

        std::string m_name;
        std::string m_timestamp;  // ISO 8601 format
        std::string m_filename;
        std::vector<JUnitProperties> m_properties;
        std::vector<JUnitTestCase> m_testcases;

        [[nodiscard]] size_t tests_count() const;
        [[nodiscard]] size_t failures_count() const;
        [[nodiscard]] size_t errors_count() const;
        [[nodiscard]] double total_time() const;
        [[nodiscard]] size_t skipped_count() const;
        [[nodiscard]] size_t assertions_count() const;
    };

    struct JUnitTestSuites
    {
        [[nodiscard]] size_t tests_count() const;
        [[nodiscard]] size_t failures_count() const;
        [[nodiscard]] size_t errors_count() const;
        [[nodiscard]] double total_time() const;
        [[nodiscard]] size_t skipped_count() const;
        [[nodiscard]] size_t assertions_count() const;

        std::string name;
        std::unordered_map<std::string, JUnitTestSuite> m_test_suites;
    };
}
