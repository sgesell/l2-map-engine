#pragma once
#include "types.hpp"
#include "bvh.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <stdexcept>

namespace l2map {

struct Node {
    NodeID id;       // 0-indexed internally
    double x, y;
    double z = 0.0;  // defaults to 0 for 2D meshes
};

struct Element {
    ElemID              id;          // 0-indexed internally
    std::string         type_name;
    std::vector<NodeID> node_ids;    // 0-indexed
};

class Mesh {
public:
    Mesh(const std::vector<Node>& nodes,
         const std::vector<Element>& elements,
         const std::string& default_element_type = "Quad8");

    const Node&    node(NodeID id) const;
    const Element& element(ElemID id) const;
    int n_nodes()    const { return static_cast<int>(nodes_.size()); }
    int n_elements() const { return static_cast<int>(elements_.size()); }

    // Global coordinates of all nodes, rows=nodes, cols=[x,y]  (2D)
    MatrixXd element_node_coords(ElemID elem_id) const;

    // Global coordinates of all nodes, rows=nodes, cols=[x,y,z]  (3D)
    MatrixXd element_node_coords_3d(ElemID elem_id) const;

    // AABB using corner nodes only (limitation: matches Python behaviour)
    std::array<double, 4> element_bbox(ElemID elem_id) const; // [xmin,xmax,ymin,ymax]

    // 3D AABB (uses z coordinate)
    AABB3D element_bbox_3d(ElemID elem_id) const;

    // Build CCW polygon for intersection testing
    Polygon2D element_polygon(ElemID elem_id) const;

    const std::vector<ElemID>& element_ids() const { return elem_id_list_; }

    const std::string& default_type() const { return default_element_type_; }

private:
    std::unordered_map<NodeID, Node>    nodes_;
    std::unordered_map<ElemID, Element> elements_;
    std::vector<ElemID>                 elem_id_list_;
    std::string                         default_element_type_;
};

} // namespace l2map
