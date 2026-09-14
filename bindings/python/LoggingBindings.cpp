#include "Bindings.h"
#include "holistic_motion/utility/Logging.h"

namespace holistic_motion::python {
namespace py = pybind11;

void BindLogging(py::module_ &module) {
    using utility::Logger;
    module.def("_set_python_logging", [](bool enable) {
        auto &logger = Logger::GetInstance();
        if (!enable) {
            logger.ResetRecordFunction();
            logger.SetVerbosityLevel(utility::VerbosityLevel::Warning);
            return;
        }
        // Capture no Python objects: singleton destruction must not DECREF after
        // interpreter shutdown. Python removes this sink in its atexit hook.
        logger.SetRecordFunction([](const utility::LogRecord &record) {
            py::gil_scoped_acquire acquire;
            py::module_::import("holistic_motion.logging")
                .attr("_emit_native")(static_cast<int>(record.level), record.file,
                                      record.line, record.function, record.message);
        });
    });
    module.def("_set_native_log_level", [](int level) {
        Logger::GetInstance().SetVerbosityLevel(
            static_cast<utility::VerbosityLevel>(level));
    });
    module.def(
        "_log_native",
        [](int level, const std::string &message, const std::string &file, int line,
           const std::string &function) {
            switch (level) {
            case 0:
                Logger::LogError_(file.c_str(), line, function.c_str(), "{}", message);
                break;
            case 1:
                Logger::LogWarning_(file.c_str(), line, function.c_str(), "{}",
                                    message);
                break;
            case 2:
                Logger::LogInfo_(file.c_str(), line, function.c_str(), "{}", message);
                break;
            case 3:
                Logger::LogDebug_(file.c_str(), line, function.c_str(), "{}", message);
                break;
            default:
                throw std::invalid_argument("native log level must be 0..3");
            }
        },
        py::arg("level"), py::arg("message"), py::arg("file") = "<native>",
        py::arg("line") = 0, py::arg("function") = "",
        py::call_guard<py::gil_scoped_release>());
}

} // namespace holistic_motion::python
