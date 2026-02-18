#include "eftx/export.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "eftx/resample.hpp"
#include "eftx/util.hpp"

namespace eftx {

static void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Could not write file: " + path);
    }
    ofs << text;
}

static std::vector<double> normalize_0_1(const std::vector<double>& in) {
    std::vector<double> out(in.size(), 0.0);
    double vmax = 0.0;
    for (std::size_t i = 0; i < in.size(); ++i) {
        double v = in[i];
        if (!is_finite(v) || v < 0.0) {
            v = 0.0;
        }
        out[i] = v;
        if (v > vmax) {
            vmax = v;
        }
    }
    if (vmax > 0.0) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] /= vmax;
        }
    }
    return out;
}

void write_pat_vertical_new_format(
    const std::string& path,
    const std::string& description,
    double gain,
    int num_antennas,
    const Pattern& vertical,
    int step_deg
) {
    const int step = step_deg > 0 ? step_deg : 1;
    Pattern v = vertical;
    v.values_lin = normalize_0_1(v.values_lin);

    std::vector<double> a_0_180;
    std::vector<double> v_0_180;
    a_0_180.reserve(v.angles_deg.size());
    v_0_180.reserve(v.values_lin.size());
    for (std::size_t i = 0; i < v.angles_deg.size(); ++i) {
        a_0_180.push_back(v.angles_deg[i] + 90.0);
        v_0_180.push_back(v.values_lin[i]);
    }

    std::vector<double> a_all = a_0_180;
    std::vector<double> v_all = v_0_180;
    for (std::size_t i = a_0_180.size(); i > 0; --i) {
        const std::size_t k = i - 1U;
        if (k == a_0_180.size() - 1U) {
            continue;
        }
        a_all.push_back(360.0 - a_0_180[k]);
        v_all.push_back(v_0_180[k]);
    }
    sort_pairs(a_all, v_all);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "'" << description << "', " << std::setprecision(2) << gain << ", " << num_antennas << "\n";
    for (int ang = 0; ang < 360; ang += step) {
        const double vv = clamp(interp_linear(static_cast<double>(ang), a_all, v_all), 0.0, 1.0);
        oss << ang << ", " << std::setprecision(4) << vv << "\n";
    }
    write_text_file(path, oss.str());
}

void write_pat_horizontal_new_format(
    const std::string& path,
    const std::string& description,
    double gain,
    int num_antennas,
    const Pattern& horizontal,
    int step_deg
) {
    const int step = step_deg > 0 ? step_deg : 1;
    std::vector<double> a(horizontal.angles_deg.size());
    std::vector<double> v = normalize_0_1(horizontal.values_lin);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = wrap_to_360(horizontal.angles_deg[i]);
    }
    sort_pairs(a, v);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "'" << description << "', " << std::setprecision(2) << gain << ", " << num_antennas << "\n";
    for (int ang = 0; ang < 360; ang += step) {
        const double vv = clamp(interp_linear(static_cast<double>(ang), a, v, true, 360.0), 0.0, 1.0);
        oss << ang << ", " << std::setprecision(4) << vv << "\n";
    }
    write_text_file(path, oss.str());
}

void write_pat_conventional_combined(
    const std::string& path,
    const std::string& description,
    double gain,
    int num_antennas,
    const Pattern& horizontal,
    const Pattern& vertical,
    int vertical_bearing_deg
) {
    Pattern h = resample_horizontal(horizontal, "none");
    Pattern v = resample_vertical_adt(vertical);
    h.values_lin = normalize_0_1(h.values_lin);
    v.values_lin = normalize_0_1(v.values_lin);

    std::vector<double> h_src_ang(h.angles_deg.size());
    for (std::size_t i = 0; i < h_src_ang.size(); ++i) {
        h_src_ang[i] = wrap_to_360(h.angles_deg[i]);
    }
    sort_pairs(h_src_ang, h.values_lin);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "'" << description << "', " << std::setprecision(2) << gain << ", " << num_antennas << "\n";
    for (int a = 0; a < 360; ++a) {
        const double vv = clamp(interp_linear(static_cast<double>(a), h_src_ang, h.values_lin, true, 360.0), 0.0, 1.0);
        oss << a << ", " << std::setprecision(4) << vv << "\n";
    }
    oss << "999\n";
    oss << "1, 91\n";
    oss << vertical_bearing_deg << ",\n";
    for (int a = 0; a >= -90; --a) {
        const double vv = clamp(interp_linear(static_cast<double>(a), v.angles_deg, v.values_lin), 0.0, 1.0);
        oss << std::setprecision(1) << static_cast<double>(a) << ", " << std::setprecision(4) << vv << "\n";
    }
    write_text_file(path, oss.str());
}

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
) {
    Pattern h = resample_horizontal(horizontal, "none");
    Pattern v = resample_vertical_adt(vertical);
    h.values_lin = normalize_0_1(h.values_lin);
    v.values_lin = normalize_0_1(v.values_lin);

    std::vector<double> h_ang(h.angles_deg.size());
    std::vector<double> h_att(h.values_lin.size());
    for (std::size_t i = 0; i < h.values_lin.size(); ++i) {
        h_ang[i] = wrap_to_360(h.angles_deg[i]);
        const double safe = std::max(h.values_lin[i], 1e-10);
        h_att[i] = std::max(-20.0 * std::log10(safe), 0.0);
    }
    sort_pairs(h_ang, h_att);

    std::vector<double> v_att(v.values_lin.size());
    for (std::size_t i = 0; i < v.values_lin.size(); ++i) {
        const double safe = std::max(v.values_lin[i], 1e-10);
        v_att[i] = std::max(-20.0 * std::log10(safe), 0.0);
    }

    std::size_t pidx = 0;
    double pmax = -1.0;
    for (std::size_t i = 0; i < v.values_lin.size(); ++i) {
        if (v.values_lin[i] > pmax) {
            pmax = v.values_lin[i];
            pidx = i;
        }
    }
    const double peak_input_angle = (pidx < v.angles_deg.size()) ? v.angles_deg[pidx] : 0.0;

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "NAME " << name << "\n";
    oss << "MAKE " << make << "\n";
    oss << "FREQUENCY " << std::setprecision(2) << frequency << " " << freq_unit << "\n";
    oss << "H_WIDTH " << std::setprecision(2) << h_width << "\n";
    oss << "V_WIDTH " << std::setprecision(2) << v_width << "\n";
    oss << "FRONT_TO_BACK " << std::setprecision(2) << front_to_back << "\n";
    oss << "GAIN " << std::setprecision(2) << gain << " dBd\n";
    oss << "TILT MECHANICAL\n";

    oss << "HORIZONTAL 360\n";
    for (int i = 0; i < 360; ++i) {
        const double val = interp_linear(static_cast<double>(i), h_ang, h_att, true, 360.0);
        oss << i << "\t" << std::setprecision(2) << val << "\n";
    }

    oss << "VERTICAL 360\n";
    for (int i = 0; i < 360; ++i) {
        const int angle_i = i % 360;
        const int diff_90 = std::abs(angle_i - 90);
        const int diff_270 = std::abs(angle_i - 270);
        const int min_diff = std::min(diff_90, diff_270);
        const double query_ang = peak_input_angle + static_cast<double>(min_diff);
        const double val = interp_linear(query_ang, v.angles_deg, v_att);
        oss << i << "\t" << std::setprecision(2) << val << "\n";
    }

    write_text_file(path, oss.str());
}

void write_pat_adt_cut(
    const std::string& path,
    const Pattern& cut,
    PatternKind kind,
    const std::string& units
) {
    Pattern r;
    if (kind == PatternKind::Horizontal) {
        r = resample_horizontal(cut, "none");
    } else if (kind == PatternKind::Vertical) {
        r = resample_vertical_adt(cut);
    } else {
        throw std::runtime_error("write_pat_adt_cut: unknown pattern kind");
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << "Edited by EFTX CPP\n";
    oss << "98\n";
    oss << "1\n";
    oss << "0 0 0 1 0\n";
    oss << units << "\n";

    for (std::size_t i = 0; i < r.angles_deg.size(); ++i) {
        oss << std::setprecision(2) << r.angles_deg[i]
            << "\t" << std::setprecision(4) << r.values_lin[i]
            << "\t0\n";
    }
    write_text_file(path, oss.str());
}

void write_table_csv(
    const std::string& path,
    const Pattern& cut,
    bool include_db
) {
    if (cut.angles_deg.size() != cut.values_lin.size()) {
        throw std::runtime_error("write_table_csv: invalid cut vectors");
    }
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    if (include_db) {
        oss << "Angle_deg,Level_linear,Level_dB\n";
    } else {
        oss << "Angle_deg,Level_linear\n";
    }
    for (std::size_t i = 0; i < cut.angles_deg.size(); ++i) {
        const double lv = std::max(cut.values_lin[i], 1e-12);
        const double ldb = 20.0 * std::log10(lv);
        oss << std::setprecision(3) << cut.angles_deg[i]
            << "," << std::setprecision(8) << cut.values_lin[i];
        if (include_db) {
            oss << "," << std::setprecision(4) << ldb;
        }
        oss << "\n";
    }
    write_text_file(path, oss.str());
}

} // namespace eftx
