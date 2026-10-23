#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezium.h"
#include "holistic_motion/utility/Logging.h"

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

using namespace holistic_motion::utility;

void Require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}

std::string Read(const std::filesystem::path &path) {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

int main() {
    auto directory =
        std::filesystem::temp_directory_path() /
        ("holistic-motion-logging-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        auto &logger = Logger::GetInstance();
        Require(logger.GetVerbosityLevel() == VerbosityLevel::Warning, "default level");
        std::atomic<int> count{0};
        logger.SetRecordFunction([&](const LogRecord &record) {
            Require(logger.GetVerbosityLevel() == VerbosityLevel::Debug,
                    "reentrant query");
            Require(record.file == "source.cpp" && record.function == "Work",
                    "source metadata");
            Require(record.line == 42 && record.message == "value 7", "record payload");
            ++count;
        });
        Logger::LogInfo_("source.cpp", 42, "Work", "value {}", 7);
        Require(count == 0, "disabled level dispatched");
        logger.SetVerbosityLevel(VerbosityLevel::Debug);
        std::vector<std::thread> workers;
        for (int i = 0; i < 4; ++i) {
            workers.emplace_back([] {
                for (int j = 0; j < 50; ++j)
                    Logger::LogInfo_("source.cpp", 42, "Work", "value {}", 7);
            });
        }
        for (auto &worker : workers)
            worker.join();
        Require(count == 200, "lost concurrent messages");
        logger.SetRecordFunction(
            [](const LogRecord &) { throw std::runtime_error("sink failed"); });
        try {
            Logger::LogError_("/tmp/source.cpp", 9, "Fail", "primary error");
        } catch (const std::runtime_error &error) {
            Require(std::string(error.what()).find("primary error") !=
                        std::string::npos,
                    "lost primary error");
            Require(std::string(error.what()).find('\x1b') == std::string::npos,
                    "ANSI in exception");
        }
        // User sinks may throw during ordinary calls, but resource cleanup
        // must not invoke them from noexcept destructors or stack unwinding.
        logger.SetRecordFunction([](const LogRecord &) {});
        using namespace holistic_motion::robotics;
        std::vector<std::shared_ptr<void>> cleanup;
        std::vector<JointNode> nodes(7);
        for (int i = 0; i < 6; ++i)
            nodes[i] = JointNode(SE3d(), Eigen::Vector3d::UnitZ(), JointType::REVOLUTE,
                                 -3.0, 3.0);
        nodes.back().joint_type = JointType::FIXED;
        cleanup.push_back(std::make_shared<NumericalKinematics>(nodes));
        cleanup.push_back(std::make_shared<OPWKinematics>(OPWParameters{}, nodes));
        cleanup.push_back(std::make_shared<URKinematics>(URParameters{}, nodes));
        using Group = Rn<double, 2>;
        std::vector<Group> points(3);
        points[0].Coeffs() << 0.0, 0.0;
        points[1].Coeffs() << 0.2, 0.1;
        points[2].Coeffs() << 0.4, 0.0;
        for (int degree : {2, 5}) {
            auto path =
                std::make_shared<PathBezierCurve<Group>>(points, degree, false, 0.01);
            auto limits = std::make_shared<TrajectoryConstraints>(
                Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones(),
                Eigen::Vector2d::Ones());
            cleanup.push_back(std::make_shared<TrajectoryDoubleS<Group>>(path, limits));
            cleanup.push_back(
                std::make_shared<TrajectoryTrapezium<Group>>(path, limits));
        }
        logger.SetRecordFunction(
            [](const LogRecord &) { throw std::runtime_error("cleanup sink failed"); });
        cleanup.clear();
        logger.SetRecordFunction([](const LogRecord &) {});
        auto foreign = std::make_shared<spdlog::logger>(
            "HOLISTIC_MOTION", std::make_shared<spdlog::sinks::null_sink_mt>());
        spdlog::register_logger(foreign);
        logger.SetRecordFunction([](const LogRecord &) {});
        std::filesystem::create_directories(directory);
        auto first = directory / "first.log";
        auto second = directory / "second.log";
        logger.SetLoggerFilePath(first.string());
        logger.EnableSaveToFile(true);
        logger.EnableSaveToFile(true);
        Logger::LogWarning_("source.cpp", 12, "First", "first marker");
        logger.SetLoggerFilePath(second.string());
        Logger::LogWarning_("source.cpp", 13, "Second", "second marker");
        // Failure must leave the current file usable.
        try {
            logger.SetLoggerFilePath(directory.string());
            throw std::logic_error("accepted a directory as a log file");
        } catch (const spdlog::spdlog_ex &) {
        }
        Logger::LogWarning_("source.cpp", 14, "Second", "after failed configuration");
        logger.EnableSaveToFile(false);
        Require(spdlog::get("HOLISTIC_MOTION") == foreign, "changed host registry");
        const auto a = Read(first), b = Read(second);
        Require(a.find("first marker") != std::string::npos &&
                    a.find("second marker") == std::string::npos,
                "path switch failed");
        Require(b.find("second marker") != std::string::npos &&
                    b.find("after failed configuration") != std::string::npos,
                "new file missing records");
        Require(b.find("Second") != std::string::npos &&
                    b.find('\x1b') == std::string::npos,
                "file metadata or encoding");
        spdlog::drop("HOLISTIC_MOTION");
        logger.ResetRecordFunction();
        logger.SetVerbosityLevel(VerbosityLevel::Warning);
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception &error) {
        Logger::GetInstance().EnableSaveToFile(false);
        Logger::GetInstance().ResetRecordFunction();
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
