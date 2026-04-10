#include "l2map/io.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdio>

namespace l2map {
namespace io {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Split a line by commas or spaces (mixed delimiters)
static std::vector<std::string> split(const std::string& line, char delim = ',') {
    std::vector<std::string> tokens;
    std::istringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, delim)) {
        std::string t = trim(tok);
        if (!t.empty()) tokens.push_back(t);
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// read_nodes
// Format: "id, x, y, z" (z ignored for Phase 1 2D)
// ---------------------------------------------------------------------------

std::vector<Node> read_nodes(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("io::read_nodes: cannot open '" + filename + "'");

    std::vector<Node> nodes;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == '*') continue;
        auto tok = split(line, ',');
        if (tok.size() < 3) continue;
        Node nd;
        nd.id = static_cast<NodeID>(std::stoi(tok[0])) - 1;  // → 0-indexed
        nd.x  = std::stod(tok[1]);
        nd.y  = std::stod(tok[2]);
        // tok[3] = z, ignored in Phase 1
        nodes.push_back(nd);
    }
    return nodes;
}

// ---------------------------------------------------------------------------
// read_elements
// Format: "id, n1, n2, ..., n8" (IDs 1-indexed in file)
// ---------------------------------------------------------------------------

std::vector<Element> read_elements(const std::string& filename,
                                   const std::string& type_name)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("io::read_elements: cannot open '" + filename + "'");

    std::vector<Element> elements;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == '*') continue;
        auto tok = split(line, ',');
        if (tok.size() < 2) continue;
        Element e;
        e.id        = static_cast<ElemID>(std::stoi(tok[0])) - 1;  // → 0-indexed
        e.type_name = type_name;
        for (size_t i = 1; i < tok.size(); ++i)
            e.node_ids.push_back(static_cast<NodeID>(std::stoi(tok[i])) - 1);
        elements.push_back(e);
    }
    return elements;
}

// ---------------------------------------------------------------------------
// read_field_data
// Format: "elem_id ipt_id v1 v2 ... vK" (space-separated)
// IDs kept as 1-indexed (MappingEngine expects 1-indexed in field_data).
// ---------------------------------------------------------------------------

MatrixXd read_field_data(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("io::read_field_data: cannot open '" + filename + "'");

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == '*') continue;
        std::istringstream ss(line);
        std::vector<double> row;
        double val;
        while (ss >> val) row.push_back(val);
        if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty()) return MatrixXd(0, 0);
    size_t ncols = rows[0].size();
    MatrixXd mat(rows.size(), ncols);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < ncols; ++j)
            mat(i, j) = (j < rows[i].size()) ? rows[i][j] : 0.0;
    }
    return mat;
}

// ---------------------------------------------------------------------------
// write_field_data
// Writes: "elem_id(1-idx) ipt_id(1-idx) v0 v1 ... vK"
// ---------------------------------------------------------------------------

void write_field_data(const std::string& filename,
                      const MappingResult& result,
                      const std::vector<ElemID>& elem_set,
                      int n_ipts_per_element,
                      const std::string& /*format*/)
{
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("io::write_field_data: cannot open '" + filename + "'");

    int n_cols = static_cast<int>(result.values.cols());
    for (int ei = 0; ei < static_cast<int>(elem_set.size()); ++ei) {
        ElemID eid_1idx = elem_set[ei] + 1;  // → 1-indexed for output
        for (int ipt = 0; ipt < n_ipts_per_element; ++ipt) {
            int row = ei * n_ipts_per_element + ipt;
            f << eid_1idx << " " << (ipt + 1);
            for (int c = 0; c < n_cols; ++c)
                f << " " << result.values(row, c);
            f << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// read_element_set
// Parses ABAQUS *ELSET, ELSET=name  sections.
// Returns 0-indexed element IDs.
// ---------------------------------------------------------------------------

std::vector<ElemID> read_element_set(const std::string& filename,
                                     const std::string& setname)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("io::read_element_set: cannot open '" + filename + "'");

    std::vector<ElemID> ids;
    bool in_set = false;
    std::string line;
    while (std::getline(f, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue;

        if (trimmed[0] == '*') {
            in_set = false;
            // Check for *ELSET, ELSET=<name>
            std::string upper = trimmed;
            std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
            if (upper.find("ELSET") != std::string::npos) {
                // Find the setname (case-insensitive)
                std::string uname = setname;
                std::transform(uname.begin(), uname.end(), uname.begin(), ::toupper);
                if (upper.find(uname) != std::string::npos)
                    in_set = true;
            }
            continue;
        }

        if (!in_set) continue;

        // Parse comma-separated IDs
        auto tok = split(trimmed, ',');
        for (const auto& t : tok) {
            if (t.empty()) continue;
            ids.push_back(static_cast<ElemID>(std::stoi(t)) - 1);
        }
    }
    return ids;
}

} // namespace io
} // namespace l2map
