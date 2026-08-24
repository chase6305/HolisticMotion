#include "holistic_motion/utility/Logging.h"
#ifdef _WIN32
#include <spdlog/fmt/bundled/printf.h>
#else
#include <fmt/printf.h>
#endif

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

namespace holistic_motion {
namespace utility {
namespace {

const char *ShortFileName(const char *file) {
    if (!file) return "<unknown>";
    const char *forward = std::strrchr(file, '/');
    const char *backward = std::strrchr(file, '\\');
    const char *separator = forward;
    if (!separator || (backward && backward > separator)) separator = backward;
    return separator ? separator + 1 : file;
}

}  // namespace

enum class TextColor {
    Black = 0,
    Red = 1,
    Green = 2,
    Yellow = 3,
    Blue = 4,
    Magenta = 5,
    Cyan = 6,
    White = 7
};

struct Logger::Impl {
    // The current print function.
    std::function<void(const std::string &)> print_fcn_;

    // The default print function (that prints to console).
    static std::function<void(const std::string &)> console_print_fcn_;

    // Verbosity level.
    VerbosityLevel verbosity_level_;

    std::shared_ptr<spdlog::logger> logger_;
    bool save_to_file_;
    std::string log_file_path_;

    // Colorize and reset the color of a string, does not work on Windows,
    std::string ColorString(const std::string &text,
                            TextColor text_color,
                            int highlight_text) const {
        std::ostringstream msg;
#ifndef _WIN32
        msg << fmt::sprintf("%c[%d;%dm", 0x1B, highlight_text,
                            (int)text_color + 30);
#endif
        msg << text;
#ifndef _WIN32
        msg << fmt::sprintf("%c[0;m", 0x1B);
#endif
        return msg.str();
    }

    static std::shared_ptr<spdlog::logger> GetOrCreateLogger(
            const std::string &log_name, const std::string &log_file_path) {
        auto lp = spdlog::get(log_name);
        if (!lp) {
            lp = spdlog::rotating_logger_mt(
                    log_name, log_file_path,
                    1048576 * DEFAULT_LOGGER_BUFFER_SIZE, 1);
        }
        return lp;
    }

    void SetSpdlogLoggerLevel(const VerbosityLevel &level) {
        if (logger_) {
            switch (level) {
                case VerbosityLevel::Error:
                    logger_->set_level(spdlog::level::err);
                    break;
                case VerbosityLevel::Warning:
                    logger_->set_level(spdlog::level::warn);
                    break;
                case VerbosityLevel::Info:
                    logger_->set_level(spdlog::level::info);
                    break;
                case VerbosityLevel::Debug:
                    logger_->set_level(spdlog::level::debug);
                    break;
                default:
                    break;
            }
        }
    }
};

#if defined(linux) || defined(__linux) || defined(__linux__)
std::function<void(const std::string &)> Logger::Impl::console_print_fcn_ =
        [](const std::string &msg) { std::cout << msg << std::endl; };

#elif defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || \
        defined(WIN64) || defined(__WIN64__) || defined(_WIN64)

#include <windows.h>

std::string utf8_to_gbk(const char *src_str) {
    int len = MultiByteToWideChar(CP_UTF8, 0, src_str, -1, NULL, 0);
    std::unique_ptr<wchar_t[]> wszGBK(new wchar_t[len + 1]);
    MultiByteToWideChar(CP_UTF8, 0, src_str, -1, wszGBK.get(), len);
    len = WideCharToMultiByte(CP_ACP, 0, wszGBK.get(), -1, NULL, 0, NULL, NULL);
    std::unique_ptr<char[]> szGBK(new char[len + 1]);
    WideCharToMultiByte(CP_ACP, 0, wszGBK.get(), -1, szGBK.get(), len, NULL,
                        NULL);
    std::string strTemp(szGBK.get());
    return strTemp;
}

std::function<void(const std::string &)> Logger::Impl::console_print_fcn_ =
        [](const std::string &msg) {
            std::cout << utf8_to_gbk(msg.c_str()) << std::endl;
        };
#endif

Logger::Logger() : impl_(new Logger::Impl()) {
    impl_->print_fcn_ = Logger::Impl::console_print_fcn_;
    impl_->verbosity_level_ = VerbosityLevel::Info;
    impl_->log_file_path_ = "log/holistic_motion.log";
    impl_->save_to_file_ = false;
}

Logger &Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::SetLoggerFilePath(const std::string &path) {
    impl_->log_file_path_ = path;
}

void Logger::EnableSaveToFile(bool enable) {
    if (enable) {
        impl_->logger_ =
                Logger::Impl::GetOrCreateLogger("HOLISTIC_MOTION", impl_->log_file_path_);
        impl_->SetSpdlogLoggerLevel(impl_->verbosity_level_);
    } else {
        impl_->logger_ = nullptr;
    }

    impl_->save_to_file_ = enable;
}

void Logger::VError [[noreturn]] (const char *file,
                                  int line,
                                  const std::string &message) const {
    file = ShortFileName(file);
    std::string err_msg =
            fmt::format("[HOLISTIC_MOTION Error] {}:{}: {}\n", file, line, message);
    err_msg = impl_->ColorString(err_msg, TextColor::Red, 1);

    if (impl_->save_to_file_ && impl_->logger_) {
        impl_->logger_->error(err_msg);
        impl_->logger_->flush();
    }

#ifdef _MSC_VER  // Uncaught exception error messages not shown in Windows
    std::cerr << err_msg << std::endl;
#endif
    throw std::runtime_error(err_msg);
}

void Logger::VWarning(const char *file,
                      int line,
                      const std::string &message) const {
    file = ShortFileName(file);
    std::string msg = fmt::format("{}:{}: {}", file, line, message);
    if (impl_->save_to_file_ && impl_->logger_) {
        impl_->logger_->warn(msg);
        impl_->logger_->flush();
    }

    std::string err_msg = fmt::format("[HOLISTIC_MOTION WARNING] {}", msg);
    err_msg = impl_->ColorString(err_msg, TextColor::Yellow, 1);

    impl_->print_fcn_(err_msg);
}

void Logger::VInfo(const char *file,
                   int line,
                   const std::string &message) const {
    file = ShortFileName(file);
    std::string msg = fmt::format("{}:{}: {}", file, line, message);
    if (impl_->save_to_file_ && impl_->logger_) {
        impl_->logger_->info(msg);
        impl_->logger_->flush();
    }

    std::string err_msg = fmt::format("[HOLISTIC_MOTION INFO] {}", msg);
    impl_->print_fcn_(err_msg);
}

void Logger::VDebug(const char *file,
                    int line,
                    const std::string &message) const {
    file = ShortFileName(file);
    std::string msg = fmt::format("{}:{}: {}", file, line, message);
    if (impl_->save_to_file_ && impl_->logger_) {
        impl_->logger_->debug(msg);
        impl_->logger_->flush();
    }

    std::string err_msg = fmt::format("[HOLISTIC_MOTION DEBUG] {}", msg);
    err_msg = impl_->ColorString(err_msg, TextColor::Cyan, 1);
    impl_->print_fcn_(err_msg);
}

void Logger::SetPrintFunction(
        std::function<void(const std::string &)> print_fcn) {
    impl_->print_fcn_ = print_fcn;
}

const std::function<void(const std::string &)> Logger::GetPrintFunction() {
    return impl_->print_fcn_;
}

void Logger::ResetPrintFunction() {
    impl_->print_fcn_ = impl_->console_print_fcn_;
}

void Logger::SetVerbosityLevel(VerbosityLevel verbosity_level) {
    impl_->verbosity_level_ = verbosity_level;
    impl_->SetSpdlogLoggerLevel(verbosity_level);
}

VerbosityLevel Logger::GetVerbosityLevel() const {
    return impl_->verbosity_level_;
}

void SetVerbosityLevel(VerbosityLevel level) {
    Logger::GetInstance().SetVerbosityLevel(level);
}

void EnableSaveToFile(bool enable) {
    Logger::GetInstance().EnableSaveToFile(enable);
}

VerbosityLevel GetVerbosityLevel() {
    return Logger::GetInstance().GetVerbosityLevel();
}

void SetLoggerFilePath(const std::string &path) {
    Logger::GetInstance().SetLoggerFilePath(path);
}

}  // namespace utility
}  // namespace holistic_motion
