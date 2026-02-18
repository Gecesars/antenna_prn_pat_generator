#pragma once

#include <string>

#include "eftx/pattern.hpp"

namespace eftx {

void write_pat_vertical_new_format(
    const std::string& path,
    const std::string& description,
    double gain,
    int num_antennas,
    const Pattern& vertical,
    int step_deg = 1
);

void write_pat_horizontal_new_format(
    const std::string& path,
    const std::string& description,
    double gain,
    int num_antennas,
    const Pattern& horizontal,
    int step_deg = 1
);

void write_pat_conventional_combined(
    const std::string& path,
    const std::string& description,
    double gain,
    int num_antennas,
    const Pattern& horizontal,
    const Pattern& vertical,
    int vertical_bearing_deg = 269
);

void write_prn_file(
    const std::string& path,
    const std::string& name,
    const std::string& make,
    double frequency,
    const std::string& freq_unit,
    double h_width,
    double v_width,
    double front_to_back,
    double gain,
    const Pattern& horizontal,
    const Pattern& vertical
);

} // namespace eftx

