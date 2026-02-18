#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace eftx {

inline bool is_finite(double v) {
    return std::isfinite(v) != 0;
}

inline double clamp(double x, double lo, double hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

inline double wrap_to_180(double a_deg) {
    double x = std::fmod(a_deg + 180.0, 360.0);
    if (x < 0.0) {
        x += 360.0;
    }
    return x - 180.0;
}

inline double wrap_to_360(double a_deg) {
    double x = std::fmod(a_deg, 360.0);
    if (x < 0.0) {
        x += 360.0;
    }
    return x;
}

double interp_linear(
    double x,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    bool periodic = false,
    double period = 360.0
);

void sort_pairs(std::vector<double>& xs, std::vector<double>& ys);

void unique_mean_by_x(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    std::vector<double>& out_xs,
    std::vector<double>& out_ys
);

std::vector<std::string> split_tokens_loose(const std::string& line);

bool try_parse_double(const std::string& s, double& out_value);

std::string trim(const std::string& s);

} // namespace eftx

