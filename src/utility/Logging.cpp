#include "holistic_motion/utility/Logging.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace holistic_motion::utility {
namespace {

const char *ShortFileName(const char *file) {
    if (!file)
        return "<unknown>";
    const char *forward = std::strrchr(file, '/');
    const char *backward = std::strrchr(file, '\\');
    const char *separator = forward;
    if (!separator || (backward && backward > separator))
        separator = backward;
    return separator ? separator + 1 : file;
}

void PrintToConsole(const std::string &message) {
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> guard(output_mutex);
    std::cout << message << std::endl;
}

spdlog::level::level_enum NativeLevel(VerbosityLevel level) {
    switch (level) {
    case VerbosityLevel::Error:
        return spdlog::level::err;
    case VerbosityLevel::Warning:
        return spdlog::level::warn;
    case VerbosityLevel::Info:
        return spdlog::level::info;
    case VerbosityLevel::Debug:
        return spdlog::level::debug;
    }
    throw std::invalid_argument("invalid verbosity level");
}

std::shared_ptr<spdlog::logger> MakeFileLogger(const std::string &path) {
    auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        path, 1048576 * DEFAULT_LOGGER_BUFFER_SIZE, 1);
    // Private logger: never take over the application's spdlog registry.
    auto logger = std::make_shared<spdlog::logger>("holistic_motion", sink);
    logger->set_level(spdlog::level::debug);
    logger->set_pattern("%Y-%m-%d %H:%M:%S.%e|%l|%n|%s-%!-%#: %v");
    return logger;
}

} // namespace

struct Logger::Impl {
    mutable std::mutex mutex;
    std::function<void(const std::string &)> print_fcn = PrintToConsole;
    std::function<void(const LogRecord &)> record_fcn;
    std::atomic<VerbosityLevel> level{VerbosityLevel::Warning};
    std::shared_ptr<spdlog::logger> file_logger;
    std::string file_path = "log/holistic_motion.log";

    void Emit(const LogRecord &record) const {
        if (record.level > level.load())
            return;
        std::function<void(const std::string &)> print;
        std::function<void(const LogRecord &)> structured;
        std::shared_ptr<spdlog::logger> file;
        {
            std::lock_guard<std::mutex> guard(mutex);
            print = print_fcn;
            structured = record_fcn;
            file = file_logger;
        }
        if (file) {
            file->log(spdlog::source_loc(record.file.c_str(), record.line,
                                         record.function.c_str()),
                      NativeLevel(record.level), "{}", record.message);
            file->flush();
        }
        if (structured) {
            structured(record);
        } else if (record.level != VerbosityLevel::Error) {
            print(fmt::format("[HOLISTIC_MOTION {}] {}:{}: {}",
                              spdlog::level::to_string_view(NativeLevel(record.level)),
                              ShortFileName(record.file.c_str()), record.line,
                              record.message));
        }
    }
};

Logger::Logger() : impl_(new Impl()) {}

Logger &Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::SetLoggerFilePath(const std::string &path) {
    if (path.empty())
        throw std::invalid_argument("logger file path must not be empty");
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (path == impl_->file_path)
        return;
    auto replacement = impl_->file_logger ? MakeFileLogger(path) : nullptr;
    impl_->file_path = path;
    impl_->file_logger = std::move(replacement);
}

void Logger::EnableSaveToFile(bool enable) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (enable && !impl_->file_logger) {
        impl_->file_logger = MakeFileLogger(impl_->file_path);
    } else if (!enable) {
        impl_->file_logger.reset();
    }
}

void Logger::VError(const char *file, int line, const char *function,
                    const std::string &message) const {
    // A logging sink must not replace the algorithm's primary error.
    try {
        impl_->Emit({VerbosityLevel::Error, file ? file : "<unknown>", line,
                     function ? function : "", message});
    } catch (...) {
    }
    throw std::runtime_error(fmt::format("[HOLISTIC_MOTION Error] {}:{}: {}",
                                         ShortFileName(file), line, message));
}

void Logger::VWarning(const char *file, int line, const char *function,
                      const std::string &message) const {
    impl_->Emit({VerbosityLevel::Warning, file ? file : "<unknown>", line,
                 function ? function : "", message});
}

void Logger::VInfo(const char *file, int line, const char *function,
                   const std::string &message) const {
    impl_->Emit({VerbosityLevel::Info, file ? file : "<unknown>", line,
                 function ? function : "", message});
}

void Logger::VDebug(const char *file, int line, const char *function,
                    const std::string &message) const {
    impl_->Emit({VerbosityLevel::Debug, file ? file : "<unknown>", line,
                 function ? function : "", message});
}

void Logger::SetPrintFunction(std::function<void(const std::string &)> print_fcn) {
    if (!print_fcn)
        throw std::invalid_argument("print function must not be empty");
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->print_fcn = std::move(print_fcn);
    impl_->record_fcn = {};
}

const std::function<void(const std::string &)> Logger::GetPrintFunction() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->print_fcn;
}

void Logger::ResetPrintFunction() { SetPrintFunction(PrintToConsole); }

void Logger::SetRecordFunction(std::function<void(const LogRecord &)> record_fcn) {
    if (!record_fcn)
        throw std::invalid_argument("record function must not be empty");
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->record_fcn = std::move(record_fcn);
}

void Logger::ResetRecordFunction() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->record_fcn = {};
}

void Logger::SetVerbosityLevel(VerbosityLevel level) {
    (void)NativeLevel(level);
    impl_->level.store(level);
}

VerbosityLevel Logger::GetVerbosityLevel() const { return impl_->level.load(); }

void SetVerbosityLevel(VerbosityLevel level) {
    Logger::GetInstance().SetVerbosityLevel(level);
}
void EnableSaveToFile(bool enable) { Logger::GetInstance().EnableSaveToFile(enable); }
VerbosityLevel GetVerbosityLevel() { return Logger::GetInstance().GetVerbosityLevel(); }
void SetLoggerFilePath(const std::string &path) {
    Logger::GetInstance().SetLoggerFilePath(path);
}

} // namespace holistic_motion::utility
