#pragma once

#include <string>
#include <utility>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace holistic_motion::python {
namespace py = pybind11;

inline void RequireRealInput(const py::handle &value, const char *name,
                             unsigned depth = 0) {
    if (PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr()))
        return;
    if (PyComplex_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must contain real values");
    }
    if (PyList_Check(value.ptr()) || PyTuple_Check(value.ptr())) {
        if (depth >= 2) {
            throw py::type_error(std::string(name) +
                                 " has an invalid array shape");
        }
        for (auto item : py::reinterpret_borrow<py::sequence>(value)) {
            RequireRealInput(item, name, depth + 1);
        }
        return;
    }
    auto array = py::isinstance<py::array>(value)
                     ? py::reinterpret_borrow<py::array>(value)
                     : py::array::ensure(value);
    if (!array) {
        throw py::type_error(std::string(name) +
                             " must contain real numeric values");
    }
    const char kind = array.dtype().kind();
    bool complex = kind == 'c';
    if (!complex && kind == 'O') {
        auto is_complex = py::module_::import("numpy").attr("iscomplexobj");
        for (auto item : array.attr("flat")) {
            if (is_complex(item).cast<bool>()) {
                complex = true;
                break;
            }
            // Object paths may contain object-typed waypoint arrays. NumPy's
            // iscomplexobj examines their dtype, not their contained values.
            if (py::isinstance<py::array>(item) || PyList_Check(item.ptr()) ||
                PyTuple_Check(item.ptr())) {
                if (depth >= 2) {
                    throw py::type_error(std::string(name) +
                                         " has an invalid array shape");
                }
                RequireRealInput(item, name, depth + 1);
            }
        }
    }
    if (complex) {
        throw py::type_error(std::string(name) + " must contain real values");
    }
}

template <typename Matrix>
Matrix CheckedRealEigen(const py::handle &value, const char *name) {
    RequireRealInput(value, name);
    try {
        return py::cast<Matrix>(value);
    } catch (const py::cast_error &) {
        throw py::type_error(std::string(name) +
                             " has an invalid array shape or type");
    }
}

template <typename Class, typename Value>
auto ReadParameter(Value Class::*member) {
    return [member](const Class &self) { return self.*member; };
}

template <typename Class, typename Value>
auto WriteRealParameter(Value Class::*member, const char *name) {
    return [member, name](Class &self, const py::object &input) {
        RequireRealInput(input, name);
        try {
            Value value = py::cast<Value>(input);
            self.*member = std::move(value);
        } catch (const py::cast_error &) {
            throw py::type_error(std::string(name) +
                                 " has an unsupported type or shape");
        }
    };
}

} // namespace holistic_motion::python
