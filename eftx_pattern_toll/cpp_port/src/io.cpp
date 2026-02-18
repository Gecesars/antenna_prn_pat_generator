#include "eftx/io.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include "eftx/util.hpp"

namespace eftx {

static std::vector<std::string> read_lines_utf8(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

Pattern parse_hfss_csv(const std::string& path) {
    const auto lines = read_lines_utf8(path);
    Pattern out;
    out.kind = PatternKind::Unknown;

    bool first = true;
    for (std::size_t li = 0; li < lines.size(); ++li) {
        std::string line = trim(lines[li]);
        if (line.empty()) {
            continue;
        }
        auto parts = split_tokens_loose(line);
        if (parts.size() < 4U) {
            continue;
        }

        if (first) {
            first = false;
            bool header_like = false;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                const std::string& p = parts[i];
                for (std::size_t j = 0; j < p.size(); ++j) {
                    const char c = p[j];
                    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                        header_like = true;
                        break;
                    }
                }
                if (header_like) {
                    break;
                }
            }
            if (header_like) {
                continue;
            }
        }

        double a = 0.0;
        double v = 0.0;
        if (try_parse_double(parts[2], a) && try_parse_double(parts.back(), v)) {
            out.angles_deg.push_back(a);
            out.values_lin.push_back(v);
        }
    }

    if (out.angles_deg.empty()) {
        throw std::runtime_error("HFSS CSV parse failed: no numeric data");
    }
    sort_pairs(out.angles_deg, out.values_lin);
    if (out.angles_deg.size() < 10U) {
        throw std::runtime_error("HFSS CSV parse failed: too few samples");
    }
    const double span = out.angles_deg.back() - out.angles_deg.front();
    if (span < 20.0) {
        throw std::runtime_error("HFSS CSV parse failed: invalid angle span");
    }
    return out;
}

Pattern parse_generic_table(const std::string& path) {
    const auto lines = read_lines_utf8(path);
    Pattern out;
    out.kind = PatternKind::Unknown;

    for (std::size_t li = 0; li < lines.size(); ++li) {
        const auto parts = split_tokens_loose(lines[li]);
        if (parts.size() < 2U) {
            continue;
        }
        std::vector<double> nums;
        nums.reserve(parts.size());
        for (std::size_t i = 0; i < parts.size(); ++i) {
            double v = 0.0;
            if (try_parse_double(parts[i], v)) {
                nums.push_back(v);
            }
        }
        if (nums.size() < 2U) {
            continue;
        }
        double angle = 0.0;
        double value = 0.0;
        if (nums.size() >= 3U) {
            // Heuristica:
            // 1) linhas ADT/PAT: "angulo valor 0" -> usa [0,1]
            // 2) linhas indexadas: "idx angulo valor" -> usa [1,2]
            const bool likely_indexed = (std::abs(nums[1]) > 30.0 && std::abs(nums[2]) <= 10.0);
            if (likely_indexed) {
                angle = nums[1];
                value = nums[2];
            } else {
                angle = nums[0];
                value = nums[1];
            }
        } else {
            angle = nums[0];
            value = nums[1];
        }
        out.angles_deg.push_back(angle);
        out.values_lin.push_back(value);
    }

    if (out.angles_deg.empty()) {
        throw std::runtime_error("Generic parse failed: no numeric data");
    }
    sort_pairs(out.angles_deg, out.values_lin);
    return out;
}

Pattern parse_auto(const std::string& path) {
    try {
        return parse_hfss_csv(path);
    } catch (...) {
        return parse_generic_table(path);
    }
}

} // namespace eftx
