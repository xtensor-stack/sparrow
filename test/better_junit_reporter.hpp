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

#include <ctime>
#include <deque>
#include <filesystem>
#include <functional>
#include <numeric>
#include <optional>

#include "junit.hpp"
#include "junit_xml_writer.hpp"

class JUnitReporter : public doctest::IReporter
{
public:

    explicit JUnitReporter(const doctest::ContextOptions& co)
        : m_options(co)
    {
    }

    // =========================================================================================
    // WHAT FOLLOWS ARE OVERRIDES OF THE VIRTUAL METHODS OF THE REPORTER INTERFACE
    // =========================================================================================

    void report_query(const doctest::QueryData&) override;
    void test_run_start() override;
    void test_run_end(const doctest::TestRunStats&) override;
    void test_case_start(const doctest::TestCaseData& in) override;
    void test_case_reenter(const doctest::TestCaseData&) override;
    void test_case_end(const doctest::CurrentTestCaseStats& c) override;
    void test_case_exception(const doctest::TestCaseException& test_case_exception) override;
    void subcase_start(const doctest::SubcaseSignature& in) override;
    void subcase_end() override;
    void log_assert(const doctest::AssertData& rb) override;
    void log_message(const doctest::MessageData& mb) override;
    void test_case_skipped(const doctest::TestCaseData&) override;

private:

    const doctest::ContextOptions& m_options;
    std::optional<std::reference_wrapper<const doctest::TestCaseData>> m_current_test_case_data;
    std::deque<std::reference_wrapper<const doctest::SubcaseSignature>> m_current_subcases_stack;
    size_t m_current_test_scope_case_failures = 0;
    size_t m_current_test_scope_assertions_count = 0;
    std::vector<sparrow::JUnitTestMessage> m_current_failures;
    std::vector<sparrow::JUnitTestMessage> m_current_errors;
    std::chrono::high_resolution_clock::time_point m_current_test_case_start_time;
    sparrow::JUnitTestSuites m_test_suites;

    sparrow::JUnitTestSuite& get_test_suite(const std::string& test_suite_name);

    void add_test_result(
        const std::string& test_suite_name,
        const std::string& classname,
        const std::string& name,
        double time,
        const std::string& file,
        size_t line,
        const std::vector<sparrow::JUnitTestMessage>& failures,
        const std::vector<sparrow::JUnitTestMessage>& errors,
        size_t assertions_count,
        bool skipped = false
    );
};

sparrow::JUnitTestSuite& JUnitReporter::get_test_suite(const std::string& test_suite_name)
{
    const auto [it, _] = m_test_suites.m_test_suites.try_emplace(
        test_suite_name,
        test_suite_name,
        doctest::JUnitReporter::JUnitTestCaseData::getCurrentTimestamp()
    );
    return it->second;
}

void JUnitReporter::add_test_result(
    const std::string& test_suite_name,
    const std::string& classname,
    const std::string& name,
    double time,
    const std::string& file,
    size_t line,
    const std::vector<sparrow::JUnitTestMessage>& failures,
    const std::vector<sparrow::JUnitTestMessage>& errors,
    size_t assertions_count,
    bool skipped
)
{
    get_test_suite(test_suite_name)
        .m_testcases.emplace_back(
            classname,
            name,
            time,
            m_options.no_path_in_filenames ? std::filesystem::path(file).filename().string() : file,
            line,
            failures,
            errors,
            assertions_count,
            skipped
        );
}

void JUnitReporter::report_query(const doctest::QueryData&)
{
}

void JUnitReporter::test_run_start()
{
    if (!m_options.minimal)
    {
        if (m_options.no_intro == false)
        {
            if (m_options.no_version == false)
            {
                std::cout << doctest::Color::Cyan << "[doctest] " << doctest::Color::None
                          << "doctest version is \"" << DOCTEST_VERSION_STR << "\"\n";
            }
            std::cout << doctest::Color::Cyan << "[doctest] " << doctest::Color::None
                      << "run with \"--" DOCTEST_OPTIONS_PREFIX_DISPLAY "help\" for options\n";
        }
    }
}

void print_failures(const sparrow::JUnitTestSuites& test_suites)
{
    for (const auto& test_suite : test_suites.m_test_suites)
    {
        for (const auto& test_case : test_suite.second.m_testcases)
        {
            if (!test_case.m_failures.empty())
            {
                std::cerr << "\tFailure(s) in file " << test_case.m_file << ":" << test_case.m_line
                          << std::endl;
                std::cerr << "\t\tClassname: " << test_case.m_classname << std::endl;
                std::cerr << "\t\tName: " << test_case.m_name << std::endl;
            }
            for (const auto& failure : test_case.m_failures)
            {
                std::cerr << "\t\tType: " << failure.m_type << std::endl;
                std::cerr << "\t\tMessage: " << failure.m_message << std::endl;
                std::cerr << "\t\tLine: " << failure.m_line << std::endl;
                std::cerr << "\t\tDetails: " << std::endl << failure.m_details << std::endl;
            }
            if (!test_case.m_failures.empty())
            {
                std::cerr << std::endl;
            }
        }
    }
}

void JUnitReporter::test_run_end(const doctest::TestRunStats&)
{
    sparrow::JUNIT_XML_Write xml_writer(*m_options.cout, m_options);
    xml_writer.write(m_test_suites);
    if (!m_options.minimal)
    {
        std::cout << "Total tests: " << m_test_suites.tests_count() << std::endl;
        std::cout << "Total assertions: " << m_test_suites.assertions_count() << std::endl;
        std::cout << "Total failures: " << m_test_suites.failures_count() << std::endl;
        print_failures(m_test_suites);
        std::cout << "Total errors: " << m_test_suites.errors_count() << std::endl;
        std::cout << "Total skipped: " << m_test_suites.skipped_count() << std::endl;
        std::cout << "Total time: " << m_test_suites.total_time() << std::endl;
        std::cout << "Export JUnit XML report: " << m_options.out << std::endl;
    }
}

void JUnitReporter::test_case_start(const doctest::TestCaseData& in)
{
    m_current_subcases_stack.clear();
    m_current_test_scope_assertions_count = 0;
    m_current_test_scope_case_failures = 0;
    m_current_failures.clear();
    m_current_errors.clear();
    m_current_test_case_data.emplace(in);
}

void JUnitReporter::test_case_reenter(const doctest::TestCaseData&)
{
}

void JUnitReporter::test_case_end(const doctest::CurrentTestCaseStats& c)
{
    add_test_result(
        m_current_test_case_data->get().m_test_suite,
        "",
        m_current_test_case_data->get().m_name,
        c.seconds,
        m_current_test_case_data->get().m_file.c_str(),
        m_current_test_case_data->get().m_line,
        m_current_failures,
        m_current_errors,
        static_cast<size_t>(c.numAssertsCurrentTest),
        m_current_test_case_data->get().m_skip
    );
}

void JUnitReporter::subcase_start(const doctest::SubcaseSignature& in)
{
    m_current_subcases_stack.emplace_back(in);
    m_current_test_scope_case_failures = 0;
    m_current_test_scope_assertions_count = 0;
    m_current_failures.clear();
    m_current_errors.clear();
    m_current_test_case_start_time = std::chrono::high_resolution_clock::now();
}

void JUnitReporter::subcase_end()
{
    const auto duration = std::chrono::high_resolution_clock::now() - m_current_test_case_start_time;
    std::string class_name;
    if (m_current_subcases_stack.size() > 1)
    {
        class_name = std::string(m_current_test_case_data->get().m_name)
                     + std::accumulate(
                         m_current_subcases_stack.begin(),
                         std::prev(m_current_subcases_stack.end()),
                         std::string(),
                         [](const std::string& a, const doctest::SubcaseSignature& b)
                         {
                             return a + "/" + std::string(b.m_name.c_str());
                         }
                     );
    }
    else
    {
        class_name = m_current_test_case_data->get().m_name;
    }

    add_test_result(
        m_current_test_case_data->get().m_test_suite,
        class_name,
        m_current_subcases_stack.back().get().m_name.c_str(),
        std::chrono::duration<double>(duration).count(),
        m_current_subcases_stack.back().get().m_file,
        static_cast<size_t>(m_current_subcases_stack.back().get().m_line),
        m_current_failures,
        m_current_errors,
        m_current_test_scope_assertions_count,
        m_current_test_case_data->get().m_skip
    );
    m_current_subcases_stack.pop_back();
}

void JUnitReporter::test_case_exception(const doctest::TestCaseException& test_case_exception)
{
    m_current_errors.emplace_back(
        "Exception",
        test_case_exception.is_crash ? "Crash" : "Exception",
        test_case_exception.error_string.c_str(),
        0
    );
}

void JUnitReporter::log_assert(const doctest::AssertData& rb)
{
    m_current_test_scope_assertions_count++;

    if (rb.m_failed)
    {
        std::ostringstream os;
        doctest::fulltext_log_assert_to_stream(os, rb);
        if (rb.m_threw)
        {
            m_current_errors
                .emplace_back(rb.m_exception_string.c_str(), rb.m_exception.c_str(), os.str(), rb.m_line);
        }
        else
        {
            m_current_failures.emplace_back("Assertion failed", os.str(), rb.m_line);
        }
    }
}

void JUnitReporter::log_message(const doctest::MessageData& mb)
{
    if (mb.m_severity & doctest::assertType::is_warn)  // report only failures
    {
        return;
    }

    std::ostringstream os;
    os << doctest::skipPathFromFilename(mb.m_file) << (m_options.gnu_file_line ? ":" : "(") << mb.m_line
       << (m_options.gnu_file_line ? ":" : "):") << std::endl;

    os << mb.m_string.c_str() << "\n";

    m_current_errors.emplace_back(
        mb.m_string.c_str(),
        mb.m_severity & doctest::assertType::is_check ? "FAIL_CHECK" : "FAIL",
        os.str(),
        mb.m_line
    );
}

void JUnitReporter::test_case_skipped(const doctest::TestCaseData&) {};

DOCTEST_REGISTER_REPORTER("better_junit", 0, JUnitReporter);
