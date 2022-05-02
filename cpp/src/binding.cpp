#include "AmplitudeComputer.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template<typename T> std::span<T> to_span(const py::buffer_info& buf) {
    return {static_cast<T*>(buf.ptr), static_cast<size_t>(buf.itemsize)};
}

PYBIND11_MODULE(mrmustard_cpp_math, m) {
    using Mustard::AmplitudeComputer;

    py::options options;
    options.disable_function_signatures();

    py::class_<AmplitudeComputer>(m, "AmplitudeComputer")
        .def(py::init<size_t>())
        .def("level_size", &AmplitudeComputer::level_size)
        .def("increase_max_n", &AmplitudeComputer::increase_max_n)
        .def("modes", &AmplitudeComputer::modes)
        .def("max_n", &AmplitudeComputer::max_n)
        .def("fill_next_level",
             [](AmplitudeComputer& ac, py::array_t<std::complex<double>, py::array::c_style> A,
                py::array_t<std::complex<double>, py::array::c_style> b,
                py::array_t<std::complex<double>, py::array::c_style> G_lower_level,
                py::array_t<std::complex<double>, py::array::c_style> G_curr_level,
                py::array_t<std::complex<double>, py::array::c_style> G_upper_level,
                size_t curr_level) -> double {
                 auto A_info = A.request();
                 auto b_info = b.request();
                 auto Gl_info = G_lower_level.request();
                 auto Gc_info = G_curr_level.request();
                 auto Gu_info = G_upper_level.request();

                 auto A_span = to_span<std::complex<double>>(A_info);
                 auto b_span = to_span<std::complex<double>>(b_info);
                 auto Gl_span = to_span<std::complex<double>>(Gl_info);
                 auto Gc_span = to_span<std::complex<double>>(Gc_info);
                 auto Gu_span = to_span<std::complex<double>>(Gu_info);
                 return ac.fill_next_level(A_span, b_span, Gl_span, Gc_span, Gu_span, curr_level);
             });
}
