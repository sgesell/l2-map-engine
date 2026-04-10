#include "l2map/bvh.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

namespace l2map {

void BVHTree2D::build(const std::vector<ElemID>& elem_ids,
                      const std::vector<AABB2D>& bboxes)
{
    nodes_.clear();
    if (elem_ids.empty()) { root_ = -1; return; }

    std::vector<int> indices(elem_ids.size());
    std::iota(indices.begin(), indices.end(), 0);
    root_ = build_recursive_(indices, bboxes, elem_ids, 0, static_cast<int>(indices.size()));
}

int BVHTree2D::build_recursive_(std::vector<int>& indices,
                                const std::vector<AABB2D>& bboxes,
                                const std::vector<ElemID>& ids,
                                int begin, int end)
{
    // Compute bounding box of all elements in [begin, end)
    AABB2D merged{
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()
    };
    for (int i = begin; i < end; ++i) {
        const AABB2D& b = bboxes[indices[i]];
        merged.xmin = std::min(merged.xmin, b.xmin);
        merged.xmax = std::max(merged.xmax, b.xmax);
        merged.ymin = std::min(merged.ymin, b.ymin);
        merged.ymax = std::max(merged.ymax, b.ymax);
    }

    int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back({merged, -1, -1, -1});

    if (end - begin == 1) {
        // Leaf node
        nodes_[node_idx].elem_id = ids[indices[begin]];
        return node_idx;
    }

    // Median split on longest axis
    double dx = merged.xmax - merged.xmin;
    double dy = merged.ymax - merged.ymin;
    int mid = begin + (end - begin) / 2;

    if (dx >= dy) {
        std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return bboxes[a].cx() < bboxes[b].cx(); });
    } else {
        std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return bboxes[a].cy() < bboxes[b].cy(); });
    }

    int left  = build_recursive_(indices, bboxes, ids, begin, mid);
    int right = build_recursive_(indices, bboxes, ids, mid, end);
    nodes_[node_idx].left  = left;
    nodes_[node_idx].right = right;
    return node_idx;
}

std::vector<ElemID> BVHTree2D::query_overlaps(const AABB2D& query_bbox) const {
    std::vector<ElemID> results;
    if (root_ < 0) return results;
    query_recursive_(root_, query_bbox, results);
    return results;
}

void BVHTree2D::query_recursive_(int node_idx,
                                 const AABB2D& query_bbox,
                                 std::vector<ElemID>& results) const
{
    const Node& n = nodes_[node_idx];
    if (!n.bbox.overlaps(query_bbox)) return;

    if (n.left == -1) {
        // Leaf
        results.push_back(n.elem_id);
        return;
    }
    query_recursive_(n.left,  query_bbox, results);
    query_recursive_(n.right, query_bbox, results);
}


// ---------------------------------------------------------------------------
// BVHTree3D
// ---------------------------------------------------------------------------

void BVHTree3D::build(const std::vector<ElemID>& elem_ids,
                      const std::vector<AABB3D>& bboxes)
{
    nodes_.clear();
    if (elem_ids.empty()) { root_ = -1; return; }

    std::vector<int> indices(elem_ids.size());
    std::iota(indices.begin(), indices.end(), 0);
    root_ = build_recursive_(indices, bboxes, elem_ids, 0, static_cast<int>(indices.size()));
}

int BVHTree3D::build_recursive_(std::vector<int>& indices,
                                const std::vector<AABB3D>& bboxes,
                                const std::vector<ElemID>& ids,
                                int begin, int end)
{
    AABB3D merged{
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()
    };
    for (int i = begin; i < end; ++i) {
        const AABB3D& b = bboxes[indices[i]];
        merged.xmin = std::min(merged.xmin, b.xmin);
        merged.xmax = std::max(merged.xmax, b.xmax);
        merged.ymin = std::min(merged.ymin, b.ymin);
        merged.ymax = std::max(merged.ymax, b.ymax);
        merged.zmin = std::min(merged.zmin, b.zmin);
        merged.zmax = std::max(merged.zmax, b.zmax);
    }

    int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back({merged, -1, -1, -1});

    if (end - begin == 1) {
        nodes_[node_idx].elem_id = ids[indices[begin]];
        return node_idx;
    }

    // Median split on longest axis
    double dx = merged.xmax - merged.xmin;
    double dy = merged.ymax - merged.ymin;
    double dz = merged.zmax - merged.zmin;
    int mid = begin + (end - begin) / 2;

    if (dx >= dy && dx >= dz) {
        std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return bboxes[a].cx() < bboxes[b].cx(); });
    } else if (dy >= dz) {
        std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return bboxes[a].cy() < bboxes[b].cy(); });
    } else {
        std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return bboxes[a].cz() < bboxes[b].cz(); });
    }

    int left  = build_recursive_(indices, bboxes, ids, begin, mid);
    int right = build_recursive_(indices, bboxes, ids, mid,   end);
    nodes_[node_idx].left  = left;
    nodes_[node_idx].right = right;
    return node_idx;
}

std::vector<ElemID> BVHTree3D::query_overlaps(const AABB3D& query_bbox) const {
    std::vector<ElemID> results;
    if (root_ < 0) return results;
    query_recursive_(root_, query_bbox, results);
    return results;
}

void BVHTree3D::query_recursive_(int node_idx,
                                 const AABB3D& query_bbox,
                                 std::vector<ElemID>& results) const
{
    const Node& n = nodes_[node_idx];
    if (!n.bbox.overlaps(query_bbox)) return;
    if (n.left == -1) {
        results.push_back(n.elem_id);
        return;
    }
    query_recursive_(n.left,  query_bbox, results);
    query_recursive_(n.right, query_bbox, results);
}

ElemID BVHTree3D::find_containing(const Point3D& p) const {
    if (root_ < 0) return -1;
    return find_containing_recursive_(root_, p);
}

ElemID BVHTree3D::find_containing_recursive_(int node_idx, const Point3D& p) const {
    const Node& n = nodes_[node_idx];
    if (!n.bbox.contains(p)) return -1;
    if (n.left == -1) return n.elem_id;  // leaf
    ElemID r = find_containing_recursive_(n.left, p);
    if (r >= 0) return r;
    return find_containing_recursive_(n.right, p);
}

} // namespace l2map
