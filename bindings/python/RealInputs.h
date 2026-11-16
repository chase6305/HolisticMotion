#pragma once

#include <string>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace holistic_motion::python {
namespace py = pybind11;

inline py::array TryMaterializeArray(const py::handle &value) {
    auto &api = py::detail::npy_api::get();
    auto array = py::reinterpret_steal<py::array>(api.PyArray_FromAny_(
        value.ptr(), nullptr, 0, 0, py::detail::npy_api::NPY_ARRAY_ENSUREARRAY_, nullptr));
    if (!array) {
        // Ordinary type/shape errors keep the caller's validation message.
        // Other Python exceptions retain their original identity.
        if (!PyErr_ExceptionMatches(PyExc_TypeError) && !PyErr_ExceptionMatches(PyExc_ValueError))
            throw py::error_already_set();
        PyErr_Clear();
    }
    return array;
}

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
    auto array = py::isinstance<py::array>(value) ? py::reinterpret_borrow<py::array>(value)
                                                  : TryMaterializeArray(value);
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
    py::object array_owner;
    py::handle checked = value;
    if constexpr (py::detail::is_eigen_dense_plain<Matrix>::value) {
        // Normalize producers and subclasses once, then check the same array
        // Eigen will read. Ordinary NumPy arrays need no conversion or copy.
        if (Py_TYPE(checked.ptr()) != py::detail::npy_api::get().PyArray_Type_) {
            array_owner = TryMaterializeArray(value);
            if (!array_owner) {
                throw py::type_error(std::string(name) + " has an invalid array shape or type");
            }
            checked = array_owner;
        }
    }
    RequireRealInput(checked, name);
    if constexpr (py::detail::is_eigen_dense_plain<Matrix>::value) {
        const auto array = py::reinterpret_borrow<py::array>(checked);
        if (array.dtype().kind() == 'O' && array.ndim() >= 1 && array.ndim() <= 2 &&
            py::detail::EigenProps<Matrix>::conformable(array)) {
            // Match Eigen's storage order while preserving object-conversion errors.
            constexpr int order = Matrix::IsRowMajor ? py::array::c_style : py::array::f_style;
            using TypedArray = py::array_t<typename Matrix::Scalar, order | py::array::forcecast>;
            try {
                array_owner = TypedArray(array);
            } catch (const py::error_already_set &error) {
                if (error.matches(PyExc_TypeError) || error.matches(PyExc_ValueError))
                    throw py::type_error(std::string(name) + " has an invalid array shape or type");
                throw;
            }
            checked = array_owner;
        }
    }
    try {
        return py::cast<Matrix>(checked);
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
        self.*member = CheckedRealEigen<Value>(input, name);
    };
}

} // namespace holistic_motion::python
