#include "eftx/project.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace eftx {

namespace {

std::string read_all_text(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Could not open project file: " + path);
    }
    std::string s;
    ifs.seekg(0, std::ios::end);
    const std::streampos end_pos = ifs.tellg();
    if (end_pos < 0) {
        throw std::runtime_error("Could not read project file size: " + path);
    }
    s.resize(static_cast<std::size_t>(end_pos));
    ifs.seekg(0, std::ios::beg);
    if (!s.empty()) {
        ifs.read(&s[0], static_cast<std::streamsize>(s.size()));
    }
    return s;
}

bool find_key_pos(const std::string& blob, const std::string& key, std::size_t& out_pos) {
    const std::string marker = "\"" + key + "\"";
    const std::size_t p = blob.find(marker);
    if (p == std::string::npos) {
        return false;
    }
    out_pos = p + marker.size();
    return true;
}

std::size_t skip_ws_colon(const std::string& blob, std::size_t p) {
    while (p < blob.size() && std::isspace(static_cast<unsigned char>(blob[p])) != 0) {
        ++p;
    }
    if (p < blob.size() && blob[p] == ':') {
        ++p;
    }
    while (p < blob.size() && std::isspace(static_cast<unsigned char>(blob[p])) != 0) {
        ++p;
    }
    return p;
}

std::string parse_json_string_at(const std::string& blob, std::size_t p) {
    if (p >= blob.size() || blob[p] != '"') {
        return std::string();
    }
    ++p;
    std::string out;
    out.reserve(64);
    bool esc = false;
    for (; p < blob.size(); ++p) {
        const char c = blob[p];
        if (esc) {
            switch (c) {
            case '"':
            case '\\':
            case '/':
                out.push_back(c);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            default:
                out.push_back(c);
                break;
            }
            esc = false;
            continue;
        }
        if (c == '\\') {
            esc = true;
            continue;
        }
        if (c == '"') {
            break;
        }
        out.push_back(c);
    }
    return out;
}

std::string extract_string_key(const std::string& blob, const std::string& key) {
    std::size_t p = 0;
    if (!find_key_pos(blob, key, p)) {
        return std::string();
    }
    p = skip_ws_colon(blob, p);
    return parse_json_string_at(blob, p);
}

std::vector<double> parse_number_list(const std::string& s) {
    std::vector<double> out;
    const char* cur = s.c_str();
    const char* end = cur + s.size();
    while (cur < end) {
        while (cur < end) {
            const char c = *cur;
            const bool number_start =
                (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.';
            if (number_start) {
                break;
            }
            ++cur;
        }
        if (cur >= end) {
            break;
        }
        char* p_end = nullptr;
        const double v = std::strtod(cur, &p_end);
        if (p_end == cur) {
            ++cur;
            continue;
        }
        out.push_back(v);
        cur = p_end;
    }
    return out;
}

std::vector<double> extract_number_array(const std::string& blob, const std::string& key) {
    std::size_t p = 0;
    if (!find_key_pos(blob, key, p)) {
        return {};
    }
    p = skip_ws_colon(blob, p);
    if (p >= blob.size()) {
        return {};
    }
    if (blob.compare(p, 4, "null") == 0) {
        return {};
    }
    if (blob[p] != '[') {
        return {};
    }
    const std::size_t start = p + 1;
    int depth = 1;
    ++p;
    for (; p < blob.size(); ++p) {
        if (blob[p] == '[') {
            ++depth;
            continue;
        }
        if (blob[p] == ']') {
            --depth;
            if (depth == 0) {
                break;
            }
        }
    }
    if (p >= blob.size() || depth != 0) {
        return {};
    }
    return parse_number_list(blob.substr(start, p - start));
}

bool assign_pattern_from_keys(
    const std::string& blob,
    const std::string& ang_key_primary,
    const std::string& val_key_primary,
    const std::string& ang_key_fallback,
    const std::string& val_key_fallback,
    PatternKind kind,
    Pattern& out
) {
    std::vector<double> a = extract_number_array(blob, ang_key_primary);
    std::vector<double> v = extract_number_array(blob, val_key_primary);
    if (a.empty() || v.empty() || a.size() != v.size()) {
        a = extract_number_array(blob, ang_key_fallback);
        v = extract_number_array(blob, val_key_fallback);
    }
    if (a.empty() || v.empty() || a.size() != v.size()) {
        return false;
    }
    out.angles_deg.swap(a);
    out.values_lin.swap(v);
    out.kind = kind;
    return true;
}

} // namespace

ProjectData load_project_json(const std::string& path) {
    const std::string blob = read_all_text(path);
    ProjectData p;
    p.base_name = extract_string_key(blob, "base_name_var");
    if (p.base_name.empty()) {
        p.base_name = "EFTX_PROJECT";
    }

    p.has_h1 = assign_pattern_from_keys(
        blob,
        "study_h1_angles",
        "study_h1_vals",
        "h_angles",
        "h_vals",
        PatternKind::Horizontal,
        p.h1
    );
    p.has_v1 = assign_pattern_from_keys(
        blob,
        "study_v1_angles",
        "study_v1_vals",
        "v_angles",
        "v_vals",
        PatternKind::Vertical,
        p.v1
    );
    p.has_h2 = assign_pattern_from_keys(
        blob,
        "study_h2_angles",
        "study_h2_vals",
        "h2_angles",
        "h2_vals",
        PatternKind::Horizontal,
        p.h2
    );
    p.has_v2 = assign_pattern_from_keys(
        blob,
        "study_v2_angles",
        "study_v2_vals",
        "v2_angles",
        "v2_vals",
        PatternKind::Vertical,
        p.v2
    );

    if (!p.has_h1 || !p.has_v1) {
        throw std::runtime_error(
            "Project JSON missing required cuts (H1/V1). Expected arrays study_h1/study_v1 or h/v."
        );
    }

    return p;
}

} // namespace eftx
