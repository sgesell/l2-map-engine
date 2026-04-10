#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "l2map/mapping_engine.hpp"
#include "l2map/mapping_engine_3d.hpp"
#include "l2map/mesh.hpp"
#include "l2map/io.hpp"

namespace py = pybind11;
// All l2map types are fully qualified to avoid conflicts with Python 3.14 C API macros.

// ---------------------------------------------------------------------------
// Convenience helper: build Mesh from numpy arrays
// nodes_arr:    shape (N, 3) or (N, 4) — [id, x, y] or [id, x, y, z]
// elements_arr: shape (M, n_nodes+1) — [id, n1, n2, ..., n_nodes]
// All IDs are 1-indexed in the arrays.
// ---------------------------------------------------------------------------

static l2map::Mesh mesh_from_numpy(
    py::array_t<double> nodes_arr,
    py::array_t<double> elements_arr,
    const std::string& element_type)
{
    auto nb = nodes_arr.unchecked<2>();
    auto eb = elements_arr.unchecked<2>();

    // Suffix _list to avoid any Python C API macro conflicts
    std::vector<l2map::Node> node_list;
    node_list.reserve(static_cast<size_t>(nb.shape(0)));
    bool has_z = (nb.shape(1) >= 4);
    for (int i = 0; i < (int)nb.shape(0); ++i) {
        l2map::Node nd;
        nd.id = static_cast<l2map::NodeID>(nb(i, 0)) - 1;  // → 0-indexed
        nd.x  = nb(i, 1);
        nd.y  = nb(i, 2);
        if (has_z) nd.z = nb(i, 3);
        node_list.push_back(nd);
    }

    std::vector<l2map::Element> elem_list;
    elem_list.reserve(static_cast<size_t>(eb.shape(0)));
    int n_node_cols = (int)eb.shape(1) - 1;
    for (int i = 0; i < (int)eb.shape(0); ++i) {
        l2map::Element e;
        e.id        = static_cast<l2map::ElemID>(eb(i, 0)) - 1;
        e.type_name = element_type;
        for (int k = 0; k < n_node_cols; ++k)
            e.node_ids.push_back(static_cast<l2map::NodeID>(eb(i, k + 1)) - 1);
        elem_list.push_back(e);
    }
    return l2map::Mesh(node_list, elem_list, element_type);
}

// ---------------------------------------------------------------------------
// High-level numpy-friendly mapping function
// ---------------------------------------------------------------------------

// Check whether an element type is 3D (registered in ElementLibrary3D).
static bool is_3d_element_type(const std::string& element_type) {
    return (element_type == "Hex8" || element_type == "Tet4");
}

static py::object map_integration_points_numpy(
    py::array_t<double> nodes_new,
    py::array_t<double> elements_new,
    py::array_t<double> nodes_old,
    py::array_t<double> elements_old,
    py::array_t<double> field_data,
    const std::string& element_type  = "Quad8",
    bool verbose          = false,
    bool enforce_positive = false,
    int  n_threads        = -1)
{
    l2map::Mesh new_mesh = mesh_from_numpy(nodes_new, elements_new, element_type);
    l2map::Mesh old_mesh = mesh_from_numpy(nodes_old, elements_old, element_type);

    // field_data: numpy array → Eigen MatrixXd
    auto fb = field_data.unchecked<2>();
    l2map::MatrixXd fd(fb.shape(0), fb.shape(1));
    for (int i = 0; i < (int)fb.shape(0); ++i)
        for (int j = 0; j < (int)fb.shape(1); ++j)
            fd(i, j) = fb(i, j);

    if (is_3d_element_type(element_type)) {
        l2map::MappingOptions3D opts3d;
        opts3d.verbose   = verbose;
        opts3d.n_threads = n_threads;

        l2map::MappingEngine3D engine3d(opts3d);
        l2map::MappingResult3D res3d = engine3d.map_integration_points(
            old_mesh, new_mesh, fd, element_type);
        return py::cast(std::move(res3d));
    }

    l2map::MappingOptions opts;
    opts.verbose          = verbose;
    opts.enforce_positive = enforce_positive;
    opts.n_threads        = n_threads;

    l2map::MappingEngine engine(opts);
    return py::cast(engine.map_integration_points(old_mesh, new_mesh, fd));
}

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(l2map_py, m) {
    m.doc() = "L2MapEngine: FEM field mapping via L2 projection";

    py::class_<l2map::MappingOptions>(m, "MappingOptions")
        .def(py::init<>())
        .def_readwrite("enforce_bounds",   &l2map::MappingOptions::enforce_bounds)
        .def_readwrite("enforce_positive", &l2map::MappingOptions::enforce_positive)
        .def_readwrite("n_gauss_pts",      &l2map::MappingOptions::n_gauss_pts)
        .def_readwrite("n_threads",        &l2map::MappingOptions::n_threads)
        .def_readwrite("verbose",          &l2map::MappingOptions::verbose);

    py::class_<l2map::MappingResult>(m, "MappingResult")
        .def_readonly("values",        &l2map::MappingResult::values)
        .def_readonly("ipoint_coords", &l2map::MappingResult::ipoint_coords)
        .def_readonly("n_clipped",     &l2map::MappingResult::n_clipped);

    py::class_<l2map::MappingResult3D>(m, "MappingResult3D")
        .def_readonly("values",        &l2map::MappingResult3D::values)
        .def_readonly("ipoint_coords", &l2map::MappingResult3D::ipoint_coords);

    m.def("map_integration_points", &map_integration_points_numpy,
          py::arg("nodes_new"),
          py::arg("elements_new"),
          py::arg("nodes_old"),
          py::arg("elements_old"),
          py::arg("field_data"),
          py::arg("element_type")     = "Quad8",
          py::arg("verbose")          = false,
          py::arg("enforce_positive") = false,
          py::arg("n_threads")        = -1,
          "Map integration point data from old mesh to new mesh via L2 projection.\n\n"
          "Parameters\n"
          "----------\n"
          "nodes_new / nodes_old : ndarray shape (N, 3+) -- [id, x, y, ...]\n"
          "elements_new / elements_old : ndarray shape (M, n_nodes+1) -- [id, n1..nN] (1-indexed)\n"
          "field_data : ndarray shape (M*n_ipts, 2+K) -- [elem_id, ipt_id, v1..vK] (1-indexed IDs)\n"
          "element_type : str, default 'Quad8'\n"
          "Returns MappingResult with .values of shape (M_new * n_ipts, K)");
}
