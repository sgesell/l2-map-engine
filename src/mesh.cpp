#include "l2map/mesh.hpp"
#include "l2map/element_library.hpp"
#include <algorithm>
#include <limits>

namespace l2map {

Mesh::Mesh(const std::vector<Node>& nodes,
           const std::vector<Element>& elements,
           const std::string& default_element_type)
    : default_element_type_(default_element_type)
{
    for (const auto& n : nodes)
        nodes_[n.id] = n;
    for (const auto& e : elements) {
        elements_[e.id] = e;
        elem_id_list_.push_back(e.id);
    }
    std::sort(elem_id_list_.begin(), elem_id_list_.end());
}

const Node& Mesh::node(NodeID id) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end())
        throw std::out_of_range("Mesh: node id " + std::to_string(id) + " not found");
    return it->second;
}

const Element& Mesh::element(ElemID id) const {
    auto it = elements_.find(id);
    if (it == elements_.end())
        throw std::out_of_range("Mesh: element id " + std::to_string(id) + " not found");
    return it->second;
}

MatrixXd Mesh::element_node_coords(ElemID elem_id) const {
    const Element& e = element(elem_id);
    int n = static_cast<int>(e.node_ids.size());
    MatrixXd coords(n, 2);
    for (int i = 0; i < n; ++i) {
        const Node& nd = node(e.node_ids[i]);
        coords(i, 0) = nd.x;
        coords(i, 1) = nd.y;
    }
    return coords;
}

MatrixXd Mesh::element_node_coords_3d(ElemID elem_id) const {
    const Element& e = element(elem_id);
    int n = static_cast<int>(e.node_ids.size());
    MatrixXd coords(n, 3);
    for (int i = 0; i < n; ++i) {
        const Node& nd = node(e.node_ids[i]);
        coords(i, 0) = nd.x;
        coords(i, 1) = nd.y;
        coords(i, 2) = nd.z;
    }
    return coords;
}

AABB3D Mesh::element_bbox_3d(ElemID elem_id) const {
    const Element& e = element(elem_id);
    int n_corners = std::min(8, static_cast<int>(e.node_ids.size()));
    AABB3D bb{
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()
    };
    for (int i = 0; i < n_corners; ++i) {
        const Node& nd = node(e.node_ids[i]);
        bb.xmin = std::min(bb.xmin, nd.x); bb.xmax = std::max(bb.xmax, nd.x);
        bb.ymin = std::min(bb.ymin, nd.y); bb.ymax = std::max(bb.ymax, nd.y);
        bb.zmin = std::min(bb.zmin, nd.z); bb.zmax = std::max(bb.zmax, nd.z);
    }
    return bb;
}

std::array<double, 4> Mesh::element_bbox(ElemID elem_id) const {
    const Element& e = element(elem_id);
    // Use only the first 4 nodes (corner nodes) to match Python reference AABB behaviour.
    int n_corners = std::min(4, static_cast<int>(e.node_ids.size()));
    double xmin =  std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin =  std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_corners; ++i) {
        const Node& nd = node(e.node_ids[i]);
        xmin = std::min(xmin, nd.x);
        xmax = std::max(xmax, nd.x);
        ymin = std::min(ymin, nd.y);
        ymax = std::max(ymax, nd.y);
    }
    return {xmin, xmax, ymin, ymax};
}

Polygon2D Mesh::element_polygon(ElemID elem_id) const {
    const Element& e = element(elem_id);
    const std::string& type = e.type_name.empty() ? default_element_type_ : e.type_name;
    MatrixXd coords = element_node_coords(elem_id);
    return ElementLibrary::instance().element_polygon(type, coords);
}

} // namespace l2map
