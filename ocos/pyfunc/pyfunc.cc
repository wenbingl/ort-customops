// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <fstream>
#include <mutex>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL ocos_python_ARRAY_API
#include <numpy/arrayobject.h>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <thread>

#include "pykernel.h"

namespace py = pybind11;

const std::map<int, int>& PyCustomOpDef::get_numpy_type_map(bool from_or_to){
  static std::map<int, int> from_type_map{
    {dt_bool, NPY_BOOL},
    {dt_float, NPY_FLOAT},
    {dt_float16, NPY_FLOAT16},
    {dt_double, NPY_DOUBLE},
    {dt_int8, NPY_INT8},
    {dt_uint8, NPY_UINT8},
    {dt_int16, NPY_INT16},
    {dt_uint16, NPY_UINT16},
    {dt_int32, NPY_INT},
    {dt_uint32, NPY_UINT},
    {dt_int64, NPY_LONGLONG},
    {dt_uint64, NPY_ULONGLONG},
  };

  static auto to_type_map = []{std::map<int, int> reversed;
                          for(auto it:from_type_map) reversed[it.second] = it.first; return reversed;}();

  return from_or_to? from_type_map: to_type_map;
}


struct PyCustomOpDefImpl: public PyCustomOpDef{

  static int to_numpy(int dt, bool from_or_to=false) {
    auto type_map = get_numpy_type_map(from_or_to);
    const auto it = type_map.find(dt);
    if (it == type_map.end()) {
      throw std::runtime_error("No corresponding Numpy data type/Tensor data Type.");
    } else {
      return it->second;
    }
  }

  typedef std::vector<int64_t> shape_t;

  static int from_numpy(int dt) {
    return to_numpy(dt, true);
  }

  template <typename _DT>
  static py::object GetPyObjFromTensor(const _DT* p, const shape_t& shape) {
    std::vector<npy_intp> npy_dims;
    for (auto n: shape) {
      npy_dims.push_back(n);
    }

    const int numpy_type = to_numpy(dt_float);
    obj = py::reinterpret_steal<py::object>(PyArray_SimpleNew(
        static_cast<int>(shape.size()), npy_dims.data(), numpy_type));

    return obj;
  }

  using callback_t = std::function<py::object(uint64_t id, py::object)>;
  static callback_t op_invoker;
};

PyCustomOpDefImpl::callback_t PyCustomOpDefImpl::op_invoker;
// static py::function g_pyfunc_caller;
// static std::mutex op_mutex;
// static std::condition_variable op_cv;
// static bool is_ready = false;

void PyCustomOpKernel::Compute(OrtKernelContext* context) {
  // std::unique_lock<std::mutex> lck(op_mutex);
  // is_ready = true;
  // op_cv.notify_all();
  //  std::this_thread::sleep_for(std::chrono::milliseconds(5000));

  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  std::vector<int64_t> dimensions;
  OrtTensorTypeAndShapeInfo* info = ort_.GetTensorTypeAndShape(input_X);
  dimensions = (ort_.GetTensorShape(info));
  ort_.ReleaseTensorTypeAndShapeInfo(info);

  py::object input0;
  PyCustomOpDefImpl::GetPyObjFromTensor(X, dimensions, input0);

  // TODO: Direct-Buffer-Access doesn't work for some reason.
  // OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  // int64_t size = ort_.GetTensorShapeElementCount(output_info);
  // ort_.ReleaseTensorTypeAndShapeInfo(output_info);
  // py::buffer_info buf(
  //     const_cast<void *>(X),                     /* Pointer to buffer */
  //     sizeof(float),                             /* Size of one scalar */
  //     py::format_descriptor<float>::format(),    /* Python struct-style format descriptor */
  //     2,                                         /* Number of dimensions */
  //     {2, 3},                                    /* Buffer dimensions */
  //     {sizeof(float) * dimensions.data()[1],     /* Strides (in bytes) for each index */
  //      sizeof(float)});

  {
    /* Acquire GIL before calling Python code */
    py::gil_scoped_acquire acquire;
    auto feed = py::make_tuple(input0);
    auto fetch = PyCustomOpDefImpl::op_invoker(obj_id_, feed);

    PyArrayObject* darray = reinterpret_cast<PyArrayObject*>(fetch.ptr());
    std::vector<int64_t> dims;
    const int npy_type = PyArray_TYPE(darray);
    {
        int ndim = PyArray_NDIM(darray);
        const npy_intp* npy_dims = PyArray_DIMS(darray);
        dims.resize(ndim);
        for (int i = 0; i < ndim; ++i) {
          dims[i] = npy_dims[i];
        }
    }

    auto element_type = PyCustomOpDefImpl::from_numpy(npy_type);
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dims.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    //TODO: A better way to calc the size of a Tensor?
    memcpy(output, PyArray_DATA(darray), sizeof(float) * [&dims](){
      size_t c=1; for (auto it = dims.begin(); it != dims.end(); ++it) c *= *it; return c;}());
  }

  // py::gil_scoped_release release;
}

const OrtCustomOp* FetchPyCustomOps(size_t& count) {
  static std::vector<PyCustomOpFactory> c_pycustomops;
  c_pycustomops.clear();

  for (auto od_ptr : PyCustomOpDef::FullList()) {
    c_pycustomops.emplace_back(PyCustomOpFactory(od_ptr));
  }

  count = c_pycustomops.size();
  return c_pycustomops.data();
}


// static std::ofstream logger;
static int init_numpy() {
  import_array();
  // logger.open("./ggtest.log.txt", std::ofstream::out | std::ofstream::app);
  // logger << "first line." << std::endl;
  return 0;
}

void AddGlobalMethods(pybind11::module& m) {
  // m.def("hook_pyfunc_caller", hook_func); // [](pybind11::function func) {
     // g_pyfunc_caller = func; });
  m.def("add_custom_op", [](const PyCustomOpDef& cod) { PyCustomOpDef::FullList().push_back(&cod); });
}

void AddObjectMethods(pybind11::module& m) {
  pybind11::class_<PyCustomOpDef>(m, "PyCustomOpDef")
      .def(pybind11::init<>())
      .def_readwrite("op_type", &PyCustomOpDef::op_type)
      .def_readwrite("obj_id", &PyCustomOpDef::obj_id)
      .def_readwrite("input_types", &PyCustomOpDef::input_types)
      .def_readwrite("output_types", &PyCustomOpDef::output_types)
      .def_static("unlock", [](){}, py::call_guard<py::gil_scoped_release>())
      .def_readwrite_static("op_invoker", &PyCustomOpDefImpl::op_invoker)
      .def_readonly_static("dt_float", &PyCustomOpDef::dt_float);
}

PYBIND11_MODULE(_ortcustomops, m) {
  m.doc() = "pybind11 stateful interface to ORT Custom Ops library";
  //RegisterExceptions(m);

  init_numpy();
  AddGlobalMethods(m);
  AddObjectMethods(m);
}
