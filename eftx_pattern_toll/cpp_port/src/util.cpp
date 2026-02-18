#include "eftx/util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace eftx {

double interp_linear(
    double x,
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    bool periodic,
    double period
) {
    if (xs.empty() || ys.empty() || xs.size() != ys.size()) {
        throw std::runtime_error("interp_linear: invalid input vectors");
    }
    if (xs.size() == 1) {
        return ys.front();
    }

    if (!periodic) {
        if (x <= xs.front()) {
            return ys.front();
        }
        if (x >= xs.back()) {
            return ys.back();
        }
        const auto it = std::lower_bound(xs.begin(), xs.end(), x);
        const std::size_t i1 = static_cast<std::size_t>(it - xs.begin());
        if (i1 == 0) {
            return ys.front();
        }
        const std::size_t i0 = i1 - 1U;
        const double x0 = xs[i0];
        const double x1 = xs[i1];
        const double y0 = ys[i0];
        const double y1 = ys[i1];
        if (x1 == x0) {
            return y0;
        }
        const double t = (x - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    const double x0 = xs.front();
    const double x1 = xs.back();
    const double span = x1 - x0;
    if (span <= 0.0 || period <= 0.0) {
        throw std::runtime_error("interp_linear periodic: invalid axis");
    }

    double xp = x;
    while (xp < x0) {
        xp += period;
    }
    while (xp > x1 + period) {
        xp -= period;
    }

    std::vector<double> xext(xs);
    std::vector<double> yext(ys);
    xext.push_back(xs.front() + period);
    yext.push_back(ys.front());

    if (xp > xext.back()) {
        xp = xext.back();
    }
    if (xp < xext.front()) {
        xp = xext.front();
    }

    const auto it = std::lower_bound(xext.begin(), xext.end(), xp);
    std::size_t i1 = static_cast<std::size_t>(it - xext.begin());
    if (i1 == 0U) {
        return yext.front();
    }
    if (i1 >= xext.size()) {
        return yext.back();
    }
    const std::size_t i0 = i1 - 1U;
    const double xa = xext[i0];
    const double xb = xext[i1];
    const double ya = yext[i0];
    const double yb = yext[i1];
    if (xb == xa) {
        return ya;
    }
    const double t = (xp - xa) / (xb - xa);
    return ya + t * (yb - ya);
}

void sort_pairs(std::vector<double>& xs, std::vector<double>& ys) {
    if (xs.size() != ys.size()) {
        throw std::runtime_error("sort_pairs: size mismatch");
    }
    std::vector<std::pair<double, double>> pairs;
    pairs.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        pairs.push_back(std::make_pair(xs[i], ys[i]));
    }
    std::sort(pairs.begin(), pairs.end(), [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
        return a.first < b.first;
    });
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        xs[i] = pairs[i].first;
        ys[i] = pairs[i].second;
    }
}

void unique_mean_by_x(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    std::vector<double>& out_xs,
    std::vector<double>& out_ys
) {
    if (xs.size() != ys.size()) {
        throw std::runtime_error("unique_mean_by_x: size mismatch");
    }
    out_xs.clear();
    out_ys.clear();
    if (xs.empty()) {
        return;
    }
    std::vector<double> x = xs;
    std::vector<double> y = ys;
    sort_pairs(x, y);

    std::size_t i = 0;
    while (i < x.size()) {
        const double xv = x[i];
        double acc = 0.0;
        std::size_t cnt = 0;
        std::size_t j = i;
        while (j < x.size() && x[j] == xv) {
            acc += y[j];
            ++cnt;
            ++j;
        }
        out_xs.push_back(xv);
        out_ys.push_back(cnt > 0 ? acc / static_cast<double>(cnt) : 0.0);
        i = j;
    }
}

std::vector<std::string> split_tokens_loose(const std::string& line) {
    std::string normalized(line);
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        char& c = normalized[i];
        if (c == '\t' || c == ';' || c == ',') {
            c = ' ';
        }
    }
    std::istringstream iss(normalized);
    std::vector<std::string> out;
    std::string token;
    while (iss >> token) {
        out.push_back(token);
    }
    return out;
}

bool try_parse_double(const std::string& s, double& out_value) {
    std::string t = trim(s);
    if (t.empty()) {
        return false;
    }
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] == ',') {
            t[i] = '.';
        }
    }
    char* endptr = nullptr;
    const double v = std::strtod(t.c_str(), &endptr);
    if (endptr == t.c_str() || *endptr != '\0') {
        return false;
    }
    if (!is_finite(v)) {
        return false;
    }
    out_value = v;
    return true;
}

std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

} // namespace eftx
