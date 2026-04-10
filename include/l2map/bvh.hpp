#pragma once
#include "types.hpp"
#include <vector>

namespace l2map {

struct AABB2D {
    double xmin, xmax, ymin, ymax;

    bool overlaps(const AABB2D& other) const {
        return xmin <= other.xmax && xmax >= other.xmin &&
               ymin <= other.ymax && ymax >= other.ymin;
    }
    bool contains(const Point2D& p) const {
        return p[0] >= xmin && p[0] <= xmax &&
               p[1] >= ymin && p[1] <= ymax;
    }
    AABB2D merge(const AABB2D& o) const {
        return {std::min(xmin, o.xmin), std::max(xmax, o.xmax),
                std::min(ymin, o.ymin), std::max(ymax, o.ymax)};
    }
    double area() const { return (xmax - xmin) * (ymax - ymin); }
    double cx()   const { return 0.5 * (xmin + xmax); }
    double cy()   const { return 0.5 * (ymin + ymax); }
};

class BVHTree2D {
public:
    void build(const std::vector<ElemID>& elem_ids,
               const std::vector<AABB2D>& bboxes);

    std::vector<ElemID> query_overlaps(const AABB2D& query_bbox) const;

    bool empty() const { return nodes_.empty(); }
    int  size()  const { return static_cast<int>(nodes_.size()); }

private:
    struct Node {
        AABB2D bbox;
        ElemID elem_id;  // -1 if internal node
        int    left;     // index into nodes_, -1 if leaf
        int    right;
    };
    std::vector<Node> nodes_;
    int               root_ = -1;

    int build_recursive_(std::vector<int>& indices,
                         const std::vector<AABB2D>& bboxes,
                         const std::vector<ElemID>& ids,
                         int begin, int end);

    void query_recursive_(int node_idx,
                          const AABB2D& query_bbox,
                          std::vector<ElemID>& results) const;
};

// ---------------------------------------------------------------------------
// 3D axis-aligned bounding box
// ---------------------------------------------------------------------------

struct AABB3D {
    double xmin, xmax, ymin, ymax, zmin, zmax;

    bool overlaps(const AABB3D& o) const {
        return xmin <= o.xmax && xmax >= o.xmin &&
               ymin <= o.ymax && ymax >= o.ymin &&
               zmin <= o.zmax && zmax >= o.zmin;
    }
    bool contains(const Point3D& p) const {
        return p[0] >= xmin && p[0] <= xmax &&
               p[1] >= ymin && p[1] <= ymax &&
               p[2] >= zmin && p[2] <= zmax;
    }
    double cx() const { return 0.5 * (xmin + xmax); }
    double cy() const { return 0.5 * (ymin + ymax); }
    double cz() const { return 0.5 * (zmin + zmax); }
};

class BVHTree3D {
public:
    void build(const std::vector<ElemID>& elem_ids,
               const std::vector<AABB3D>& bboxes);

    std::vector<ElemID> query_overlaps(const AABB3D& query_bbox) const;

    // Return the single element whose AABB contains p (first match, or -1)
    ElemID find_containing(const Point3D& p) const;

    bool empty() const { return nodes_.empty(); }
    int  size()  const { return static_cast<int>(nodes_.size()); }

private:
    struct Node {
        AABB3D bbox;
        ElemID elem_id;  // -1 if internal node
        int    left;
        int    right;
    };
    std::vector<Node> nodes_;
    int               root_ = -1;

    int build_recursive_(std::vector<int>& indices,
                         const std::vector<AABB3D>& bboxes,
                         const std::vector<ElemID>& ids,
                         int begin, int end);

    void query_recursive_(int node_idx,
                          const AABB3D& query_bbox,
                          std::vector<ElemID>& results) const;

    ElemID find_containing_recursive_(int node_idx, const Point3D& p) const;
};

} // namespace l2map
