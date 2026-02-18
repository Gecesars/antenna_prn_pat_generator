#include "eftx/metrics.hpp"

#include <cmath>

#include "eftx/util.hpp"

namespace eftx {

double simpson_uniform(const std::vector<double>& y, double dx) {
    const std::size_t n = y.size();
    if (n < 2U) {
        return 0.0;
    }
    if ((n % 2U) == 0U) {
        std::vector<double> head(y.begin(), y.end() - 1);
        return simpson_uniform(head, dx) + 0.5 * dx * (y[n - 2] + y[n - 1]);
    }
    double s = y.front() + y.back();
    for (std::size_t i = 1; i + 1 < n; i += 2) {
        s += 4.0 * y[i];
    }
    for (std::size_t i = 2; i + 1 < n; i += 2) {
        s += 2.0 * y[i];
    }
    return s * dx / 3.0;
}

double hpbw_deg(const Pattern& in) {
    if (in.angles_deg.size() < 3U || in.angles_deg.size() != in.values_lin.size()) {
        return std::nan("");
    }
    double vmax = 0.0;
    for (std::size_t i = 0; i < in.values_lin.size(); ++i) {
        if (in.values_lin[i] > vmax) {
            vmax = in.values_lin[i];
        }
    }
    if (vmax <= 0.0) {
        return std::nan("");
    }

    const double thr = std::sqrt(0.5);
    std::vector<double> e(in.values_lin.size());
    std::size_t i0 = 0;
    double peak = -1.0;
    for (std::size_t i = 0; i < in.values_lin.size(); ++i) {
        e[i] = in.values_lin[i] / vmax;
        if (e[i] > peak) {
            peak = e[i];
            i0 = i;
        }
    }

    bool has_left = false;
    bool has_right = false;
    double a_left = 0.0;
    double a_right = 0.0;

    for (std::size_t i = i0; i > 0; --i) {
        if (e[i] >= thr && e[i - 1] < thr) {
            const double x1 = in.angles_deg[i - 1];
            const double x2 = in.angles_deg[i];
            const double y1 = e[i - 1];
            const double y2 = e[i];
            const double den = (y2 - y1);
            a_left = (std::abs(den) < 1e-12) ? x1 : (x1 + (thr - y1) * (x2 - x1) / den);
            has_left = true;
            break;
        }
    }
    for (std::size_t i = i0; i + 1 < e.size(); ++i) {
        if (e[i] >= thr && e[i + 1] < thr) {
            const double x1 = in.angles_deg[i];
            const double x2 = in.angles_deg[i + 1];
            const double y1 = e[i];
            const double y2 = e[i + 1];
            const double den = (y2 - y1);
            a_right = (std::abs(den) < 1e-12) ? x1 : (x1 + (thr - y1) * (x2 - x1) / den);
            has_right = true;
            break;
        }
    }

    if (!has_left || !has_right) {
        return std::nan("");
    }
    return a_right - a_left;
}

double directivity_2d_cut(const Pattern& in, double span_deg) {
    if (in.angles_deg.size() < 2U || in.angles_deg.size() != in.values_lin.size()) {
        return std::nan("");
    }
    double vmax = 0.0;
    for (std::size_t i = 0; i < in.values_lin.size(); ++i) {
        if (in.values_lin[i] > vmax) {
            vmax = in.values_lin[i];
        }
    }
    if (vmax <= 0.0) {
        return std::nan("");
    }

    std::vector<double> p(in.values_lin.size());
    for (std::size_t i = 0; i < in.values_lin.size(); ++i) {
        const double e = in.values_lin[i] / vmax;
        p[i] = e * e;
    }

    const double dx = (in.angles_deg[1] - in.angles_deg[0]) * (3.14159265358979323846 / 180.0);
    const double integral = simpson_uniform(p, dx);
    if (integral <= 0.0) {
        return std::nan("");
    }
    const double span_rad = span_deg * (3.14159265358979323846 / 180.0);
    return span_rad / integral;
}

} // namespace eftx

