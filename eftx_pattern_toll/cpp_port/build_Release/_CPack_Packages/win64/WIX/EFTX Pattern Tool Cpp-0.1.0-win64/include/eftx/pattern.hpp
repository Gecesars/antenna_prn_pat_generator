#pragma once

#include <string>
#include <vector>

namespace eftx {

enum class PatternKind {
    Unknown = 0,
    Horizontal = 1,
    Vertical = 2,
};

struct Pattern {
    std::vector<double> angles_deg;
    std::vector<double> values_lin;
    PatternKind kind = PatternKind::Unknown;
    std::string name;
};

inline const char* kind_to_string(PatternKind k) {
    switch (k) {
    case PatternKind::Horizontal:
        return "H";
    case PatternKind::Vertical:
        return "V";
    default:
        return "U";
    }
}

inline PatternKind kind_from_string(const std::string& s) {
    if (s == "H" || s == "h" || s == "HRP" || s == "hrp") {
        return PatternKind::Horizontal;
    }
    if (s == "V" || s == "v" || s == "VRP" || s == "vrp") {
        return PatternKind::Vertical;
    }
    return PatternKind::Unknown;
}

} // namespace eftx

