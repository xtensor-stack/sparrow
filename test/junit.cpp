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

#include "junit.hpp"

#include <numeric>

namespace sparrow
{

    JUnitTestMessage::JUnitTestMessage(std::string message, std::string type, std::string details, size_t line)
        : m_message(std::move(message))
        , m_type(std::move(type))
        , m_details(std::move(details))
        , m_line(line)
    {
    }

    JUnitTestMessage::JUnitTestMessage(std::string message, std::string details, size_t line)
        : m_message(std::move(message))
        , m_details(std::move(details))
        , m_line(line)
    {
    }

    JUnitTestCase::JUnitTestCase(
        std::string classname,
        std::string name,
        double time,
        std::string file,
        size_t line,
        std::vector<JUnitTestMessage> failures,
        std::vector<JUnitTestMessage> errors,
        size_t assertions_count,
        bool skipped
    )
        : m_classname(std::move(classname))
        , m_name(std::move(name))
        , m_time(time)
        , m_file(std::move(file))
        , m_line(line)
        , m_failures(std::move(failures))
        , m_errors(std::move(errors))
        , m_assertions_count(assertions_count)
        , m_skipped(skipped)
    {
    }

    size_t JUnitTestSuite::tests_count() const
    {
        return m_testcases.size();
    }

    [[nodiscard]] size_t JUnitTestSuite::failures_count() const
    {
        return std::accumulate(
            m_testcases.begin(),
            m_testcases.end(),
            size_t(0),
            [](size_t count, const JUnitTestCase& testcase)
            {
                return count + testcase.m_failures.size();
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuite::errors_count() const
    {
        return std::accumulate(
            m_testcases.begin(),
            m_testcases.end(),
            size_t(0),
            [](size_t count, const JUnitTestCase& testcase)
            {
                return count + testcase.m_errors.size();
            }
        );
    }

    [[nodiscard]] double JUnitTestSuite::total_time() const
    {
        return std::accumulate(
            m_testcases.begin(),
            m_testcases.end(),
            0.,
            [](double time, const JUnitTestCase& testcase)
            {
                return time + testcase.m_time;
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuite::skipped_count() const
    {
        return std::accumulate(
            m_testcases.begin(),
            m_testcases.end(),
            size_t(0),
            [](size_t count, const JUnitTestCase& testcase)
            {
                return count + (testcase.m_skipped ? 1 : 0);
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuite::assertions_count() const
    {
        return std::accumulate(
            m_testcases.begin(),
            m_testcases.end(),
            size_t(0),
            [](size_t count, const JUnitTestCase& testcase)
            {
                return count + testcase.m_assertions_count;
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuites::tests_count() const
    {
        return std::accumulate(
            m_test_suites.begin(),
            m_test_suites.end(),
            size_t(0),
            [](size_t count, const std::pair<std::string, JUnitTestSuite>& testsuite)
            {
                return count + testsuite.second.tests_count();
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuites::failures_count() const
    {
        return std::accumulate(
            m_test_suites.begin(),
            m_test_suites.end(),
            size_t(0),
            [](size_t count, const std::pair<std::string, JUnitTestSuite>& testsuite)
            {
                return count + testsuite.second.failures_count();
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuites::errors_count() const
    {
        return std::accumulate(
            m_test_suites.begin(),
            m_test_suites.end(),
            size_t(0),
            [](size_t count, const std::pair<std::string, JUnitTestSuite>& testsuite)
            {
                return count + testsuite.second.errors_count();
            }
        );
    }

    [[nodiscard]] double JUnitTestSuites::total_time() const
    {
        return std::accumulate(
            m_test_suites.begin(),
            m_test_suites.end(),
            0.,
            [](double time, const std::pair<std::string, JUnitTestSuite>& testsuite)
            {
                return time + testsuite.second.total_time();
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuites::skipped_count() const
    {
        return std::accumulate(
            m_test_suites.begin(),
            m_test_suites.end(),
            size_t(0),
            [](size_t count, const std::pair<std::string, JUnitTestSuite>& testsuite)
            {
                return count + testsuite.second.skipped_count();
            }
        );
    }

    [[nodiscard]] size_t JUnitTestSuites::assertions_count() const
    {
        return std::accumulate(
            m_test_suites.begin(),
            m_test_suites.end(),
            size_t(0),
            [](size_t count, const std::pair<std::string, JUnitTestSuite>& testsuite)
            {
                return count + testsuite.second.assertions_count();
            }
        );
    }

}
