#pragma once

#include <functional>
#include <memory>
#include <string>

// NVCC does not support deprecated attribute on Windows prior to v11.
#if defined(__CUDACC__) && defined(_MSC_VER) && __CUDACC_VER_MAJOR__ < 11
#ifndef FMT_DEPRECATED
#define FMT_DEPRECATED
#endif
#endif

#include <spdlog/fmt/fmt.h>

// The maximum size of a logging file. (default: 10MB)
#define DEFAULT_LOGGER_BUFFER_SIZE 10

#if defined(_MSC_VER)
#define HOLISTIC_MOTION_FUNCTION __FUNCSIG__
#else
#define HOLISTIC_MOTION_FUNCTION __PRETTY_FUNCTION__
#endif

// Mimic "macro in namespace" by concatenating `utility::` and a macro.
// Ref: https://stackoverflow.com/a/11791202
//
// We avoid using (format, ...) since in this case __VA_ARGS__ can be
// empty, and the behavior of pruning trailing comma with ##__VA_ARGS__ is not
// officially standard.
// Ref: https://stackoverflow.com/a/28074198
//
// __PRETTY_FUNCTION__ has to be converted, otherwise a bug regarding [noreturn]
// will be triggered.
// Ref: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94742

// LogError throws now a runtime_error with the given error message. This
// should be used if there is no point in continuing the given algorithm at
// some point and the error is not returned in another way (e.g., via a
// bool/int as return value).
//
// Usage  : utility::LogError(format_string, arg0, arg1, ...);
// Example: utility::LogError("name: {}, age: {}", "dog", 5);
#define LogError(...)                     \
    Logger::LogError_(__FILE__, __LINE__, \
                      static_cast<const char *>(HOLISTIC_MOTION_FUNCTION), __VA_ARGS__)

// LogWarning is used if an error occurs, but the error is also signaled
// via a return value (i.e., there is no need to throw an exception). This
// warning should further be used, if the algorithms encounters a state
// that does not break its continuation, but the output is likely not to be
// what the user expected.
//
// Usage  : utility::LogWarning(format_string, arg0, arg1, ...);
// Example: utility::LogWarning("name: {}, age: {}", "dog", 5);
#define LogWarning(...)                                             \
    Logger::LogWarning_(__FILE__, __LINE__,                         \
                        static_cast<const char *>(HOLISTIC_MOTION_FUNCTION), \
                        __VA_ARGS__)

// LogInfo is used to inform the user with expected output, e.g, pressed a
// key in the visualizer prints helping information.
//
// Usage  : utility::LogInfo(format_string, arg0, arg1, ...);
// Example: utility::LogInfo("name: {}, age: {}", "dog", 5);
#define LogInfo(...)                     \
    Logger::LogInfo_(__FILE__, __LINE__, \
                     static_cast<const char *>(HOLISTIC_MOTION_FUNCTION), __VA_ARGS__)

// LogDebug is used to print debug/additional information on the state of
// the algorithm.
//
// Usage  : utility::LogDebug(format_string, arg0, arg1, ...);
// Example: utility::LogDebug("name: {}, age: {}", "dog", 5);
#define LogDebug(...)                     \
    Logger::LogDebug_(__FILE__, __LINE__, \
                      static_cast<const char *>(HOLISTIC_MOTION_FUNCTION), __VA_ARGS__)

namespace holistic_motion {
namespace utility {

enum class VerbosityLevel {
    /// LogError throws now a runtime_error with the given error message. This
    /// should be used if there is no point in continuing the given algorithm at
    /// some point and the error is not returned in another way (e.g., via a
    /// bool/int as return value).
    Error = 0,
    /// LogWarning is used if an error occurs, but the error is also signaled
    /// via a return value (i.e., there is no need to throw an exception). This
    /// warning should further be used, if the algorithms encounters a state
    /// that does not break its continuation, but the output is likely not to be
    /// what the user expected.
    Warning = 1,
    /// LogInfo is used to inform the user with expected output, e.g, pressed a
    /// key in the visualizer prints helping information.
    Info = 2,
    /// LogDebug is used to print debug/additional information on the state of
    /// the algorithm.
    Debug = 3,
};

/// Logger class should be used as a global singleton object (GetInstance()).
class Logger {
public:
    Logger(Logger const &) = delete;
    void operator=(Logger const &) = delete;

    /// Get Logger global singleton instance.
    static Logger &GetInstance();

    /// Overwrite the default print function, this is useful when you want to
    /// redirect prints rather than printing to stdout. For example, in HOLISTIC_MOTION's
    /// python binding, the default print function is replaced with py::print().
    ///
    /// \param print_fcn The function for printing. It should take a string
    /// input and returns nothing.
    void SetPrintFunction(std::function<void(const std::string &)> print_fcn);

    /// Reset the print function to the default one (print to console).
    void ResetPrintFunction();

    /// Get the print function used by the Logger.
    const std::function<void(const std::string &)> GetPrintFunction();

    /// Set global verbosity level of HOLISTIC_MOTION.
    ///
    /// \param verbosity_level Messages with equal or less than verbosity_level
    /// verbosity will be printed.
    void SetVerbosityLevel(VerbosityLevel verbosity_level);

    /// Get global verbosity level of HOLISTIC_MOTION.
    VerbosityLevel GetVerbosityLevel() const;

    /// Set Logger file path.
    void SetLoggerFilePath(const std::string &path);

    /// Set to save logging info to file.
    void EnableSaveToFile(bool enable);

    template <typename... Args>
    static void LogError_ [[noreturn]] (const char *file,
                                        int line,
                                        const char *function,
                                        const char *format,
                                        Args &&... args) {
        (void)function;
        std::string message;
        if (sizeof...(Args) > 0) {
            message = FormatArgs(format, fmt::make_format_args(args...));
        } else {
            message = std::string(format);
        }
        Logger::GetInstance().VError(file, line, message);
    }
    template <typename... Args>
    static void LogWarning_(const char *file,
                            int line,
                            const char *function,
                            const char *format,
                            Args &&... args) {
        (void)function;
        if (Logger::GetInstance().GetVerbosityLevel() >=
            VerbosityLevel::Warning) {
            std::string message;
            if (sizeof...(Args) > 0) {
                message = FormatArgs(format, fmt::make_format_args(args...));
            } else {
                message = std::string(format);
            }
            Logger::GetInstance().VWarning(file, line, message);
        }
    }
    template <typename... Args>
    static void LogInfo_(const char *file,
                         int line,
                         const char *function,
                         const char *format,
                         Args &&... args) {
        (void)function;
        if (Logger::GetInstance().GetVerbosityLevel() >= VerbosityLevel::Info) {
            std::string message;
            if (sizeof...(Args) > 0) {
                message = FormatArgs(format, fmt::make_format_args(args...));
            } else {
                message = std::string(format);
            }
            Logger::GetInstance().VInfo(file, line, message);
        }
    }
    template <typename... Args>
    static void LogDebug_(const char *file,
                          int line,
                          const char *function,
                          const char *format,
                          Args &&... args) {
        (void)function;
        if (Logger::GetInstance().GetVerbosityLevel() >=
            VerbosityLevel::Debug) {
            std::string message;
            if (sizeof...(Args) > 0) {
                message = FormatArgs(format, fmt::make_format_args(args...));
            } else {
                message = std::string(format);
            }
            Logger::GetInstance().VDebug(file, line, message);
        }
    }

private:
    Logger();
    static std::string FormatArgs(const char *format, fmt::format_args args) {
        std::string err_msg = fmt::vformat(format, args);
        return err_msg;
    }
    void VError [[noreturn]] (const char *file,
                              int line,
                              const std::string &message) const;
    void VWarning(const char *file, int line, const std::string &message) const;
    void VInfo(const char *file, int line, const std::string &message) const;
    void VDebug(const char *file, int line, const std::string &message) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Set global verbosity level of HOLISTIC_MOTION
///
/// \param level Messages with equal or less than verbosity_level verbosity will
/// be printed.
void SetVerbosityLevel(VerbosityLevel level);

/// Get global verbosity level of HOLISTIC_MOTION.
VerbosityLevel GetVerbosityLevel();

/// Enable save to file flag globally.
/// \param enable If true, a log file will be created in the global log file
/// path.
void EnableSaveToFile(bool enable);

/// Set logger file path.
void SetLoggerFilePath(const std::string &path);

}  // namespace utility
}  // namespace holistic_motion
