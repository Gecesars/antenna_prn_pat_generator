#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "eftx/export.hpp"
#include "eftx/io.hpp"
#include "eftx/metrics.hpp"
#include "eftx/pattern.hpp"
#include "eftx/project.hpp"
#include "eftx/dashboard.hpp"
#include "eftx/resample.hpp"

namespace {

void print_help() {
    std::cout
        << "EFTX Pattern Tool (C++ core)\n"
        << "Usage:\n"
        << "  eftx_pattern_tool_cpp <command> [options]\n\n"
        << "Commands:\n"
        << "  metrics --in <file> --kind <H|V>\n"
        << "  resample --in <file> --kind <H|V> --out <file> [--norm none|max|rms]\n"
        << "  export-pat-h --in <file> --out <file> [--desc txt] [--gain g] [--num n] [--step s]\n"
        << "  export-pat-v --in <file> --out <file> [--desc txt] [--gain g] [--num n] [--step s]\n"
        << "  export-pat-combined --h <file> --v <file> --out <file> [--desc txt] [--gain g] [--num n]\n"
        << "  export-prn --h <file> --v <file> --out <file> [--name n] [--make m] [--freq f] [--funit MHz]\n"
        << "  project-export --project <file.eftxproj.json> --out <dir> [--norm none|max|rms] (gera dashboard HTML)\n"
        << "\n";
}

std::map<std::string, std::string> parse_kv(int argc, char** argv, int start) {
    std::map<std::string, std::string> kv;
    for (int i = start; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) {
            continue;
        }
        if (i + 1 < argc) {
            kv[k] = argv[++i];
        } else {
            kv[k] = "";
        }
    }
    return kv;
}

std::string getv(const std::map<std::string, std::string>& kv, const std::string& k, const std::string& d = "") {
    const std::map<std::string, std::string>::const_iterator it = kv.find(k);
    if (it == kv.end()) {
        return d;
    }
    return it->second;
}

double getd(const std::map<std::string, std::string>& kv, const std::string& k, double d) {
    const std::string s = getv(kv, k, "");
    if (s.empty()) {
        return d;
    }
    return std::atof(s.c_str());
}

int geti(const std::map<std::string, std::string>& kv, const std::string& k, int d) {
    const std::string s = getv(kv, k, "");
    if (s.empty()) {
        return d;
    }
    return std::atoi(s.c_str());
}

void write_two_col(const std::string& path, const eftx::Pattern& p) {
    std::ofstream ofs(path.c_str(), std::ios::out | std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Could not write: " + path);
    }
    ofs.setf(std::ios::fixed);
    for (std::size_t i = 0; i < p.angles_deg.size(); ++i) {
        ofs << p.angles_deg[i] << "\t" << p.values_lin[i] << "\n";
    }
}

eftx::Pattern load_kinded(const std::string& path, eftx::PatternKind kind) {
    eftx::Pattern p = eftx::parse_auto(path);
    p.kind = kind;
    return p;
}

std::string path_join(const std::string& a, const std::string& b) {
    const std::filesystem::path p = std::filesystem::path(a) / std::filesystem::path(b);
    return p.string();
}

std::string filename_only(const std::string& p) {
    return std::filesystem::path(p).filename().string();
}

struct ExportedPolInfo {
    int pol = 1;
    eftx::Pattern h;
    eftx::Pattern v;

    std::string az_pat;
    std::string el_pat;
    std::string az_adt;
    std::string el_adt;
    std::string az_csv;
    std::string el_csv;
    std::string az_svg;
    std::string el_svg;
    std::string prn;
};

ExportedPolInfo export_pol_artifacts(
    const std::string& out_dir,
    const std::string& base_name,
    int pol,
    const eftx::Pattern& h_in,
    const eftx::Pattern& v_in,
    const std::string& norm
) {
    ExportedPolInfo info;
    info.pol = pol;

    eftx::Pattern h = eftx::resample_horizontal(h_in, norm);
    eftx::Pattern v = eftx::resample_vertical_adt(v_in);
    if (norm != "none") {
        v = eftx::normalize_pattern(v, norm);
    }
    info.h = h;
    info.v = v;

    const std::string pol_s = std::to_string(pol);
    info.az_pat = path_join(out_dir, base_name + "_AZ_POL" + pol_s + ".pat");
    info.el_pat = path_join(out_dir, base_name + "_EL_POL" + pol_s + ".pat");
    info.az_adt = path_join(out_dir, base_name + "_AZ_POL" + pol_s + "_ADT.pat");
    info.el_adt = path_join(out_dir, base_name + "_EL_POL" + pol_s + "_ADT.pat");
    info.az_csv = path_join(out_dir, base_name + "_AZ_POL" + pol_s + "_TABELA.csv");
    info.el_csv = path_join(out_dir, base_name + "_EL_POL" + pol_s + "_TABELA.csv");
    info.az_svg = path_join(out_dir, base_name + "_AZ_POL" + pol_s + "_DIAGRAMA.svg");
    info.el_svg = path_join(out_dir, base_name + "_EL_POL" + pol_s + "_DIAGRAMA.svg");
    info.prn = path_join(out_dir, base_name + "_POL" + pol_s + ".prn");

    eftx::write_pat_horizontal_new_format(
        info.az_pat,
        base_name + " AZ_POL" + pol_s,
        0.0,
        1,
        h,
        1
    );
    eftx::write_pat_vertical_new_format(
        info.el_pat,
        base_name + " EL_POL" + pol_s,
        0.0,
        1,
        v,
        1
    );
    eftx::write_pat_adt_cut(info.az_adt, h, eftx::PatternKind::Horizontal);
    eftx::write_pat_adt_cut(info.el_adt, v, eftx::PatternKind::Vertical);
    eftx::write_table_csv(info.az_csv, h, true);
    eftx::write_table_csv(info.el_csv, v, true);
    eftx::write_prn_file(info.prn, base_name + "_POL" + pol_s, "EFTX", 99.5, "MHz", 65.0, 45.0, 25.0, 0.0, h, v);
    eftx::write_cut_svg(info.az_svg, h, eftx::PatternKind::Horizontal, base_name + " | POL" + pol_s + " | Azimute");
    eftx::write_cut_svg(info.el_svg, v, eftx::PatternKind::Vertical, base_name + " | POL" + pol_s + " | Elevacao");

    std::cout << "POL" << pol << " exported:\n";
    std::cout << "  " << info.az_pat << "\n";
    std::cout << "  " << info.el_pat << "\n";
    std::cout << "  " << info.az_adt << "\n";
    std::cout << "  " << info.el_adt << "\n";
    std::cout << "  " << info.az_csv << "\n";
    std::cout << "  " << info.el_csv << "\n";
    std::cout << "  " << info.az_svg << "\n";
    std::cout << "  " << info.el_svg << "\n";
    std::cout << "  " << info.prn << "\n";
    return info;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_help();
            return 0;
        }
        const std::string cmd = argv[1];
        if (cmd == "--help" || cmd == "-h" || cmd == "help") {
            print_help();
            return 0;
        }

        const std::map<std::string, std::string> kv = parse_kv(argc, argv, 2);

        if (cmd == "metrics") {
            const std::string in = getv(kv, "--in");
            const eftx::PatternKind kind = eftx::kind_from_string(getv(kv, "--kind"));
            if (in.empty() || kind == eftx::PatternKind::Unknown) {
                throw std::runtime_error("metrics: --in and --kind <H|V> are required");
            }
            eftx::Pattern p = load_kinded(in, kind);
            if (kind == eftx::PatternKind::Horizontal) {
                p = eftx::resample_horizontal(p, "max");
            } else {
                p = eftx::resample_vertical(p, "max");
            }
            const double hpbw = eftx::hpbw_deg(p);
            const double d2d = eftx::directivity_2d_cut(p, kind == eftx::PatternKind::Horizontal ? 360.0 : 180.0);
            std::cout << "HPBW(deg): " << hpbw << "\n";
            std::cout << "D2D(lin): " << d2d << "\n";
            if (d2d > 0.0) {
                std::cout << "D2D(dB): " << (10.0 * std::log10(d2d)) << "\n";
            }
            return 0;
        }

        if (cmd == "resample") {
            const std::string in = getv(kv, "--in");
            const std::string out = getv(kv, "--out");
            const std::string norm = getv(kv, "--norm", "none");
            const eftx::PatternKind kind = eftx::kind_from_string(getv(kv, "--kind"));
            if (in.empty() || out.empty() || kind == eftx::PatternKind::Unknown) {
                throw std::runtime_error("resample: --in --out --kind <H|V> are required");
            }
            eftx::Pattern p = load_kinded(in, kind);
            eftx::Pattern r = (kind == eftx::PatternKind::Horizontal) ? eftx::resample_horizontal(p, norm) : eftx::resample_vertical(p, norm);
            write_two_col(out, r);
            std::cout << "OK: " << out << " (" << r.angles_deg.size() << " pts)\n";
            return 0;
        }

        if (cmd == "export-pat-h") {
            const std::string in = getv(kv, "--in");
            const std::string out = getv(kv, "--out");
            if (in.empty() || out.empty()) {
                throw std::runtime_error("export-pat-h: --in and --out are required");
            }
            eftx::Pattern p = load_kinded(in, eftx::PatternKind::Horizontal);
            eftx::Pattern r = eftx::resample_horizontal(p, "none");
            eftx::write_pat_horizontal_new_format(
                out,
                getv(kv, "--desc", "HRP_cpp"),
                getd(kv, "--gain", 0.0),
                geti(kv, "--num", 1),
                r,
                geti(kv, "--step", 1)
            );
            std::cout << "OK: " << out << "\n";
            return 0;
        }

        if (cmd == "export-pat-v") {
            const std::string in = getv(kv, "--in");
            const std::string out = getv(kv, "--out");
            if (in.empty() || out.empty()) {
                throw std::runtime_error("export-pat-v: --in and --out are required");
            }
            eftx::Pattern p = load_kinded(in, eftx::PatternKind::Vertical);
            eftx::Pattern r = eftx::resample_vertical_adt(p);
            eftx::write_pat_vertical_new_format(
                out,
                getv(kv, "--desc", "VRP_cpp"),
                getd(kv, "--gain", 0.0),
                geti(kv, "--num", 1),
                r,
                geti(kv, "--step", 1)
            );
            std::cout << "OK: " << out << "\n";
            return 0;
        }

        if (cmd == "export-pat-combined") {
            const std::string h = getv(kv, "--h");
            const std::string v = getv(kv, "--v");
            const std::string out = getv(kv, "--out");
            if (h.empty() || v.empty() || out.empty()) {
                throw std::runtime_error("export-pat-combined: --h --v --out are required");
            }
            eftx::Pattern hp = load_kinded(h, eftx::PatternKind::Horizontal);
            eftx::Pattern vp = load_kinded(v, eftx::PatternKind::Vertical);
            eftx::write_pat_conventional_combined(
                out,
                getv(kv, "--desc", "PAT_combined_cpp"),
                getd(kv, "--gain", 0.0),
                geti(kv, "--num", 1),
                hp,
                vp,
                geti(kv, "--bearing", 269)
            );
            std::cout << "OK: " << out << "\n";
            return 0;
        }

        if (cmd == "export-prn") {
            const std::string h = getv(kv, "--h");
            const std::string v = getv(kv, "--v");
            const std::string out = getv(kv, "--out");
            if (h.empty() || v.empty() || out.empty()) {
                throw std::runtime_error("export-prn: --h --v --out are required");
            }
            eftx::Pattern hp = load_kinded(h, eftx::PatternKind::Horizontal);
            eftx::Pattern vp = load_kinded(v, eftx::PatternKind::Vertical);
            eftx::write_prn_file(
                out,
                getv(kv, "--name", "ANTENA_CPP"),
                getv(kv, "--make", "EFTX"),
                getd(kv, "--freq", 99.5),
                getv(kv, "--funit", "MHz"),
                getd(kv, "--hwidth", 65.0),
                getd(kv, "--vwidth", 45.0),
                getd(kv, "--fb", 25.0),
                getd(kv, "--gain", 0.0),
                hp,
                vp
            );
            std::cout << "OK: " << out << "\n";
            return 0;
        }

        if (cmd == "project-export") {
            const std::string proj = getv(kv, "--project");
            const std::string out = getv(kv, "--out");
            const std::string norm = getv(kv, "--norm", "none");
            if (proj.empty() || out.empty()) {
                throw std::runtime_error("project-export: --project and --out are required");
            }

            std::filesystem::create_directories(std::filesystem::path(out));
            const eftx::ProjectData pdata = eftx::load_project_json(proj);
            std::vector<ExportedPolInfo> exported;
            exported.push_back(export_pol_artifacts(out, pdata.base_name, 1, pdata.h1, pdata.v1, norm));
            if (pdata.has_h2 && pdata.has_v2) {
                exported.push_back(export_pol_artifacts(out, pdata.base_name, 2, pdata.h2, pdata.v2, norm));
            }

            std::vector<eftx::DashboardCut> cuts;
            std::vector<std::string> extras;
            for (std::size_t i = 0; i < exported.size(); ++i) {
                const ExportedPolInfo& e = exported[i];
                const std::string pol_s = std::to_string(e.pol);

                eftx::DashboardCut az;
                az.title = pdata.base_name + " | POL" + pol_s + " | Azimute";
                az.kind = eftx::PatternKind::Horizontal;
                az.pattern = e.h;
                az.image_relpath = filename_only(e.az_svg);
                az.table_relpath = filename_only(e.az_csv);
                az.pat_relpath = filename_only(e.az_pat);
                az.adt_relpath = filename_only(e.az_adt);
                cuts.push_back(az);

                eftx::DashboardCut el;
                el.title = pdata.base_name + " | POL" + pol_s + " | Elevacao";
                el.kind = eftx::PatternKind::Vertical;
                el.pattern = e.v;
                el.image_relpath = filename_only(e.el_svg);
                el.table_relpath = filename_only(e.el_csv);
                el.pat_relpath = filename_only(e.el_pat);
                el.adt_relpath = filename_only(e.el_adt);
                cuts.push_back(el);

                extras.push_back(filename_only(e.prn));
            }

            const std::string dashboard_html = path_join(out, pdata.base_name + "_dashboard.html");
            eftx::write_project_dashboard(dashboard_html, pdata.base_name, cuts, extras);
            std::cout << "Dashboard: " << dashboard_html << "\n";
            std::cout << "OK: project export finished\n";
            return 0;
        }

        throw std::runtime_error("Unknown command: " + cmd);
    } catch (const std::exception& e) {
        std::cerr << "[ERR] " << e.what() << "\n";
        return 1;
    }
}
