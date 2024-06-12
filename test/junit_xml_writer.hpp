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

#define DOCTEST_CONFIG_IMPLEMENT
#include <numeric>

#include <doctest/doctest.h>

#include "junit.hpp"

namespace sparrow
{
    class JUNIT_XML_Write
    {
    public:

        JUNIT_XML_Write(std::ostream& s, const doctest::ContextOptions& opt);

        void write(const JUnitTestMessage& message, const std::string& type);
        void write(const JUnitTestCase& test_case);
        void write(const std::vector<JUnitProperties>& properties);
        void write(const JUnitTestSuite& test_suite);
        void write(const JUnitTestSuites& test_suites);

    private:

        doctest::XmlWriter m_xml_writer;
        const doctest::ContextOptions& m_context_options;
    };
}

namespace sparrow
{
    JUNIT_XML_Write::JUNIT_XML_Write(std::ostream& s, const doctest::ContextOptions& opt)
        : m_xml_writer(s)
        , m_context_options(opt)
    {
        m_xml_writer.writeDeclaration();
    }

    void JUNIT_XML_Write::write(const JUnitTestMessage& message, const std::string& type)
    {
        m_xml_writer.startElement(type);

        if (!message.m_message.empty())
        {
            m_xml_writer.writeAttribute("message", message.m_message);
        }
        if (!message.m_type.empty())
        {
            m_xml_writer.writeAttribute("type", message.m_type);
        }
        if(message.m_line != 0)
        {
            m_xml_writer.writeAttribute("line", message.m_line);
        }
        if (!message.m_details.empty())
        {
            m_xml_writer.writeText(message.m_details, false);
        }

        m_xml_writer.endElement();
    }

    void JUNIT_XML_Write::write(const JUnitTestCase& test_case)
    {
        m_xml_writer.startElement("testcase")
            .writeAttribute("classname", test_case.m_classname)
            .writeAttribute("name", test_case.m_name)
            .writeAttribute("line", test_case.m_line)
            .writeAttribute("assertions", test_case.m_assertions_count)
            .writeAttribute("skipped", test_case.m_skipped)
            .writeAttribute("file", test_case.m_file);

        if (!m_context_options.no_time_in_output)
        {
            m_xml_writer.writeAttribute("time", test_case.m_time);
        }

        for (const auto& failure : test_case.m_failures)
        {
            write(failure, "failure");
        }

        for (const auto& error : test_case.m_errors)
        {
            write(error, "error");
        }

        if (test_case.m_skipped)
        {
            m_xml_writer.scopedElement("skipped").writeAttribute("message", "Test was skipped.");
        }

        m_xml_writer.endElement();
    }

    void JUNIT_XML_Write::write(const std::vector<JUnitProperties>& properties)
    {
        m_xml_writer.startElement("properties");

        for (const auto& property : properties)
        {
            if (property.m_value.size() == 1)
            {
                m_xml_writer.scopedElement("property")
                    .writeAttribute("name", property.m_name)
                    .writeAttribute("value", property.m_value[0]);
            }
            else
            {
                const std::string text = std::accumulate(
                    property.m_value.begin(),
                    property.m_value.end(),
                    std::string(),
                    [](const std::string& a, const std::string& b)
                    {
                        return a + b + "\n";
                    }
                );
                m_xml_writer.scopedElement("property").writeAttribute("name", property.m_name).writeText(text);
            }
        }

        m_xml_writer.endElement();
    }

    void JUNIT_XML_Write::write(const JUnitTestSuite& test_suite)
    {
        m_xml_writer.startElement("testsuite")
            .writeAttribute("name", test_suite.m_name)
            .writeAttribute("tests", test_suite.tests_count())
            .writeAttribute("errors", test_suite.errors_count())
            .writeAttribute("failures", test_suite.failures_count())
            .writeAttribute("skipped", test_suite.skipped_count());

        if (!m_context_options.no_time_in_output)
        {
            m_xml_writer.writeAttribute("time", test_suite.total_time())
                .writeAttribute("timestamp", test_suite.m_timestamp);
        }

        if (!test_suite.m_properties.empty())
        {
            write(test_suite.m_properties);
        }

        for (const auto& test_case : test_suite.m_testcases)
        {
            write(test_case);
        }

        m_xml_writer.endElement();
    }

    void JUNIT_XML_Write::write(const JUnitTestSuites& test_suites)
    {
        m_xml_writer.startElement("testsuites")
            .writeAttribute("name", test_suites.name)
            .writeAttribute("tests", test_suites.tests_count())
            .writeAttribute("errors", test_suites.errors_count())
            .writeAttribute("failures", test_suites.failures_count())
            .writeAttribute("skipped", test_suites.skipped_count())
            .writeAttribute("assertions", test_suites.assertions_count())
            .writeAttribute("time", test_suites.total_time());

        for (const auto& [_ , test_suite] : test_suites.m_test_suites)
        {
            write(test_suite);
        }

        m_xml_writer.endElement();
    }
}
