#include "eftx/resample.hpp"

#include <cmath>
#include <stdexcept>

#include "eftx/util.hpp"

namespace eftx {

static std::vector<double> normalize_values(const std::vector<double>& values, const std::string& mode) {
    if (mode == "none") {
        return values;
    }
    std::vector<double> out(values);
    if (out.empty()) {
        return out;
    }

    if (mode == "max") {
        double vmax = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (out[i] > vmax) {
                vmax = out[i];
            }
        }
        if (vmax <= 0.0) {
            vmax = 1.0;
        }
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] /= vmax;
        }
        return out;
    }

    if (mode == "rms") {
        double acc = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            acc += out[i] * out[i];
        }
        double rms = std::sqrt(acc / static_cast<double>(out.size()));
        if (rms <= 0.0) {
            rms = 1.0;
        }
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] /= rms;
        }
    }
    return out;
}

Pattern normalize_pattern(const Pattern& in, const std::string& mode) {
    Pattern out = in;
    out.values_lin = normalize_values(in.values_lin, mode);
    return out;
}

Pattern resample_vertical(const Pattern& in, const std::string& norm) {
    if (in.angles_deg.size() != in.values_lin.size() || in.angles_deg.empty()) {
        throw std::runtime_error("resample_vertical: invalid input");
    }
    std::vector<double> a;
    std::vector<double> v;
    a.reserve(in.angles_deg.size());
    v.reserve(in.values_lin.size());

    for (std::size_t i = 0; i < in.angles_deg.size(); ++i) {
        const double ang = in.angles_deg[i];
        if (ang >= -90.0 && ang <= 90.0 && is_finite(ang) && is_finite(in.values_lin[i])) {
            a.push_back(ang);
            v.push_back(in.values_lin[i]);
        }
    }
    if (a.empty()) {
        throw std::runtime_error("vertical data does not cover [-90, 90]");
    }

    std::vector<double> ua;
    std::vector<double> uv;
    unique_mean_by_x(a, v, ua, uv);
    if (ua.empty()) {
        throw std::runtime_error("vertical unique axis is empty");
    }

    Pattern out;
    out.kind = PatternKind::Vertical;
    for (int i = 0; i <= 1800; ++i) {
        const double t = -90.0 + 0.1 * static_cast<double>(i);
        out.angles_deg.push_back(std::round(t * 10.0) / 10.0);
        out.values_lin.push_back(interp_linear(t, ua, uv));
    }
    out.values_lin = normalize_values(out.values_lin, norm);
    return out;
}

Pattern resample_vertical_adt(const Pattern& in) {
    try {
        return resample_vertical(in, "none");
    } catch (...) {
    }
    Pattern shifted = in;
    for (std::size_t i = 0; i < shifted.angles_deg.size(); ++i) {
        shifted.angles_deg[i] -= 90.0;
    }
    try {
        return resample_vertical(shifted, "none");
    } catch (...) {
    }
    Pattern wrapped = in;
    for (std::size_t i = 0; i < wrapped.angles_deg.size(); ++i) {
        wrapped.angles_deg[i] = wrap_to_180(wrapped.angles_deg[i]);
    }
    return resample_vertical(wrapped, "none");
}

Pattern resample_horizontal(const Pattern& in, const std::string& norm) {
    if (in.angles_deg.size() != in.values_lin.size() || in.angles_deg.empty()) {
        throw std::runtime_error("resample_horizontal: invalid input");
    }

    std::vector<double> a(in.angles_deg.size());
    std::vector<double> v(in.values_lin);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = wrap_to_180(in.angles_deg[i]);
    }
    sort_pairs(a, v);
    if (a.size() < 2U) {
        throw std::runtime_error("resample_horizontal: insufficient data");
    }

    std::vector<double> a_ext(a);
    std::vector<double> v_ext(v);
    a_ext.push_back(a.front() + 360.0);
    v_ext.push_back(v.front());

    Pattern out;
    out.kind = PatternKind::Horizontal;
    for (int deg = -180; deg <= 180; ++deg) {
        const double t = static_cast<double>(deg);
        double t_adj = t;
        if (t_adj < a_ext.front()) {
            t_adj += 360.0;
        }
        out.angles_deg.push_back(t);
        out.values_lin.push_back(interp_linear(t_adj, a_ext, v_ext));
    }
    out.values_lin = normalize_values(out.values_lin, norm);
    return out;
}

} // namespace eftx

