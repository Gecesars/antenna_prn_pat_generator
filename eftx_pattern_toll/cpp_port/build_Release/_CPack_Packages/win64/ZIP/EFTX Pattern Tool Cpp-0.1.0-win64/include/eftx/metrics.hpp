#pragma once

#include "eftx/pattern.hpp"

namespace eftx {

double simpson_uniform(const std::vector<double>& y, double dx);
double hpbw_deg(const Pattern& in);
double directivity_2d_cut(const Pattern& in, double span_deg);

} // namespace eftx

