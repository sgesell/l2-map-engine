#pragma once
#include "types.hpp"
#include "mesh.hpp"
#include "mapping_engine.hpp"
#include <string>
#include <vector>

namespace l2map {
namespace io {

// Node file format: "id, x, y, z" (comma-separated, one node per line)
// IDs are 1-indexed in file; returned nodes are 0-indexed internally.
std::vector<Node> read_nodes(const std::string& filename);

// Element file format: "id, n1, n2, ..., n8" (comma-separated)
// IDs are 1-indexed in file; returned elements are 0-indexed internally.
std::vector<Element> read_elements(const std::string& filename,
                                   const std::string& type_name = "Quad8");

// Field data file format: "elem_id ipt_id v1 v2 ... vK" (space-separated)
// IDs remain 1-indexed (as expected by MappingEngine.build_field_cache_).
MatrixXd read_field_data(const std::string& filename);

// Write mapped result in same format as field data input
void write_field_data(const std::string& filename,
                      const MappingResult& result,
                      const std::vector<ElemID>& elem_set,
                      int n_ipts_per_element,
                      const std::string& format = "%.11f");

// Parse an ABAQUS *ELSET section from a file.
// Returns 0-indexed element IDs.
std::vector<ElemID> read_element_set(const std::string& filename,
                                     const std::string& setname);

} // namespace io
} // namespace l2map
