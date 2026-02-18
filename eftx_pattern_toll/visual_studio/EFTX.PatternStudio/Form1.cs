using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace EFTX.PatternStudio;

public partial class Form1 : Form
{
    private const double PlotFloorDb = -40.0;
    private static readonly Regex NumberRegex = new(@"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?", RegexOptions.Compiled);

    private List<PointPair> _hrpRaw = [];
    private List<PointPair> _vrpRaw = [];
    private List<PointPair> _hrpDb = [];
    private List<PointPair> _vrpDb = [];
    private bool _busy;

    private readonly record struct PointPair(double Angle, double Value);

    public Form1()
    {
        InitializeComponent();
        cmbNorm.SelectedIndex = 0;
        txtOutput.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "EFTX_Exports");
        txtCoreExe.Text = FindDefaultCoreExe();
        EnableDoubleBuffer(pnlHrp);
        EnableDoubleBuffer(pnlVrp);
        Log("Studio visual inicializado.");
    }

    private static void EnableDoubleBuffer(Control control)
    {
        var prop = typeof(Control).GetProperty("DoubleBuffered", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        prop?.SetValue(control, true, null);
    }

    private static string FindDefaultCoreExe()
    {
        var probes = new List<string>();
        var root = AppContext.BaseDirectory;
        var dir = new DirectoryInfo(root);
        for (var i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            probes.Add(Path.Combine(dir.FullName, "cpp_port", "build_cpp", "Release", "eftx_pattern_tool_cpp.exe"));
            probes.Add(Path.Combine(dir.FullName, "eftx_pattern_toll", "cpp_port", "build_cpp", "Release", "eftx_pattern_tool_cpp.exe"));
            probes.Add(Path.Combine(dir.FullName, "cpp_port", "build_visual", "Release", "eftx_pattern_tool_cpp.exe"));
            probes.Add(Path.Combine(dir.FullName, "eftx_pattern_toll", "cpp_port", "build_visual", "Release", "eftx_pattern_tool_cpp.exe"));
        }
        foreach (var p in probes)
        {
            if (File.Exists(p))
            {
                return p;
            }
        }
        return string.Empty;
    }

    private void Log(string message)
    {
        if (InvokeRequired)
        {
            BeginInvoke(() => Log(message));
            return;
        }
        txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}");
    }

    private void SetBusy(bool busy)
    {
        _busy = busy;
        btnBuildCore.Enabled = !busy;
        btnPreview.Enabled = !busy;
        btnExport.Enabled = !busy;
        btnBrowseCoreExe.Enabled = !busy;
        btnBrowseHrp.Enabled = !busy;
        btnBrowseVrp.Enabled = !busy;
        btnBrowseOutput.Enabled = !busy;
    }

    private void btnBrowseCoreExe_Click(object sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Filter = "Executavel (*.exe)|*.exe|Todos os arquivos (*.*)|*.*",
            Title = "Selecionar core C++"
        };
        if (dlg.ShowDialog(this) == DialogResult.OK)
        {
            txtCoreExe.Text = dlg.FileName;
        }
    }

    private void btnBrowseHrp_Click(object sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Filter = "Arquivos de diagrama (*.pat;*.txt;*.csv)|*.pat;*.txt;*.csv|Todos os arquivos (*.*)|*.*",
            Title = "Selecionar arquivo de azimute"
        };
        if (dlg.ShowDialog(this) == DialogResult.OK)
        {
            txtHrp.Text = dlg.FileName;
            if (string.IsNullOrWhiteSpace(txtBaseName.Text))
            {
                txtBaseName.Text = Path.GetFileNameWithoutExtension(dlg.FileName);
            }
        }
    }

    private void btnBrowseVrp_Click(object sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Filter = "Arquivos de diagrama (*.pat;*.txt;*.csv)|*.pat;*.txt;*.csv|Todos os arquivos (*.*)|*.*",
            Title = "Selecionar arquivo de elevacao"
        };
        if (dlg.ShowDialog(this) == DialogResult.OK)
        {
            txtVrp.Text = dlg.FileName;
            if (string.IsNullOrWhiteSpace(txtBaseName.Text))
            {
                txtBaseName.Text = Path.GetFileNameWithoutExtension(dlg.FileName);
            }
        }
    }

    private void btnBrowseOutput_Click(object sender, EventArgs e)
    {
        using var dlg = new FolderBrowserDialog
        {
            Description = "Selecionar pasta de saida"
        };
        if (dlg.ShowDialog(this) == DialogResult.OK)
        {
            txtOutput.Text = dlg.SelectedPath;
        }
    }

    private void btnOpenOutput_Click(object sender, EventArgs e)
    {
        try
        {
            if (!Directory.Exists(txtOutput.Text))
            {
                Directory.CreateDirectory(txtOutput.Text);
            }
            Process.Start(new ProcessStartInfo
            {
                FileName = txtOutput.Text,
                UseShellExecute = true
            });
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Falha ao abrir pasta", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private async void btnBuildCore_Click(object sender, EventArgs e)
    {
        if (_busy)
        {
            return;
        }

        var cppRoot = ResolveCppRoot();
        if (cppRoot == null)
        {
            MessageBox.Show(this, "Nao foi possivel localizar cpp_port.", "Build Core", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var buildDir = Path.Combine(cppRoot, "build_vs_gui");
        Directory.CreateDirectory(buildDir);

        SetBusy(true);
        try
        {
            Log("Configurando CMake para Visual Studio 2022...");
            var c1 = await RunProcessAsync("cmake", ["-S", cppRoot, "-B", buildDir, "-G", "Visual Studio 17 2022", "-A", "x64"], cppRoot);
            if (c1 != 0)
            {
                throw new InvalidOperationException("Falha na configuracao CMake.");
            }
            Log("Compilando core C++ (Release)...");
            var c2 = await RunProcessAsync("cmake", ["--build", buildDir, "--config", "Release", "--target", "eftx_pattern_tool_cpp"], cppRoot);
            if (c2 != 0)
            {
                throw new InvalidOperationException("Falha no build do core C++.");
            }

            var exe = Path.Combine(buildDir, "Release", "eftx_pattern_tool_cpp.exe");
            if (File.Exists(exe))
            {
                txtCoreExe.Text = exe;
                Log("Core compilado com sucesso.");
            }
            else
            {
                Log("Build finalizado, mas o EXE nao foi encontrado no caminho esperado.");
            }
        }
        catch (Exception ex)
        {
            Log("[ERR] " + ex.Message);
            MessageBox.Show(this, ex.Message, "Build Core", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private string? ResolveCppRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 10 && dir != null; i++, dir = dir.Parent)
        {
            var c1 = Path.Combine(dir.FullName, "cpp_port", "CMakeLists.txt");
            if (File.Exists(c1))
            {
                return Path.GetDirectoryName(c1);
            }
            var c2 = Path.Combine(dir.FullName, "eftx_pattern_toll", "cpp_port", "CMakeLists.txt");
            if (File.Exists(c2))
            {
                return Path.GetDirectoryName(c2);
            }
        }
        return null;
    }

    private async void btnExport_Click(object sender, EventArgs e)
    {
        if (_busy)
        {
            return;
        }

        if (!TryLoadCuts(showDialog: true))
        {
            return;
        }
        if (!File.Exists(txtCoreExe.Text))
        {
            MessageBox.Show(this, "Core C++ nao encontrado. Compile com 'Build Core' ou selecione o EXE.", "Exportacao", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        var outDir = txtOutput.Text.Trim();
        if (string.IsNullOrWhiteSpace(outDir))
        {
            MessageBox.Show(this, "Informe a pasta de saida.", "Exportacao", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        Directory.CreateDirectory(outDir);
        var baseName = string.IsNullOrWhiteSpace(txtBaseName.Text) ? "Projeto_VisualStudio" : SanitizeName(txtBaseName.Text.Trim());
        var projectFile = Path.Combine(outDir, $"{baseName}.eftxproj.json");

        SetBusy(true);
        try
        {
            WriteProjectJson(projectFile, baseName, _hrpRaw, _vrpRaw);
            Log("Projeto JSON gerado: " + projectFile);

            var norm = (cmbNorm.SelectedItem?.ToString() ?? "none").Trim();
            Log("Executando core C++ para exportar artefatos...");
            var code = await RunProcessAsync(
                txtCoreExe.Text,
                ["project-export", "--project", projectFile, "--out", outDir, "--norm", norm],
                Path.GetDirectoryName(txtCoreExe.Text) ?? outDir
            );
            if (code != 0)
            {
                throw new InvalidOperationException("Core retornou erro na exportacao.");
            }

            var dashboard = Path.Combine(outDir, $"{baseName}_dashboard.html");
            if (File.Exists(dashboard))
            {
                Log("Dashboard gerado: " + dashboard);
                Process.Start(new ProcessStartInfo { FileName = dashboard, UseShellExecute = true });
            }
            MessageBox.Show(this, "Exportacao concluida com sucesso.", "Exportacao", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
        catch (Exception ex)
        {
            Log("[ERR] " + ex.Message);
            MessageBox.Show(this, ex.Message, "Exportacao", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void btnPreview_Click(object sender, EventArgs e)
    {
        if (TryLoadCuts(showDialog: true))
        {
            Log($"Preview atualizado. AZ={_hrpRaw.Count} pts, EL={_vrpRaw.Count} pts.");
        }
    }

    private bool TryLoadCuts(bool showDialog)
    {
        try
        {
            var hrpPath = txtHrp.Text.Trim();
            var vrpPath = txtVrp.Text.Trim();
            if (!File.Exists(hrpPath))
            {
                throw new FileNotFoundException("Arquivo de azimute nao encontrado.", hrpPath);
            }
            if (!File.Exists(vrpPath))
            {
                throw new FileNotFoundException("Arquivo de elevacao nao encontrado.", vrpPath);
            }

            _hrpRaw = ParsePatternFile(hrpPath);
            _vrpRaw = ParsePatternFile(vrpPath);
            _hrpDb = ToNormalizedDb(_hrpRaw);
            _vrpDb = ToNormalizedDb(_vrpRaw);

            UpdateMetricsGrid();
            pnlHrp.Invalidate();
            pnlVrp.Invalidate();
            return true;
        }
        catch (Exception ex)
        {
            if (showDialog)
            {
                MessageBox.Show(this, ex.Message, "Carga de diagramas", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            Log("[ERR] " + ex.Message);
            return false;
        }
    }

    private static List<PointPair> ParsePatternFile(string path)
    {
        var outPts = new List<PointPair>(1024);
        var firstLine = true;
        foreach (var raw in File.ReadLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0)
            {
                continue;
            }

            if (firstLine && line.Any(char.IsLetter))
            {
                firstLine = false;
                continue;
            }
            firstLine = false;

            var nums = ExtractNumbers(line);
            if (nums.Count < 2)
            {
                continue;
            }

            double angle;
            double val;
            if (nums.Count >= 4)
            {
                angle = nums[2];
                val = nums[^1];
            }
            else if (nums.Count >= 3)
            {
                var likelyIndexed = Math.Abs(nums[1]) > 30.0 && Math.Abs(nums[2]) <= 10.0;
                if (likelyIndexed)
                {
                    angle = nums[1];
                    val = nums[2];
                }
                else
                {
                    angle = nums[0];
                    val = nums[1];
                }
            }
            else
            {
                angle = nums[0];
                val = nums[1];
            }

            if (double.IsFinite(angle) && double.IsFinite(val))
            {
                outPts.Add(new PointPair(angle, val));
            }
        }

        if (outPts.Count == 0)
        {
            throw new InvalidOperationException("Nenhum dado numerico valido foi encontrado.");
        }
        outPts.Sort((a, b) => a.Angle.CompareTo(b.Angle));
        return outPts;
    }

    private static List<double> ExtractNumbers(string line)
    {
        var list = new List<double>(8);
        foreach (Match m in NumberRegex.Matches(line))
        {
            var s = m.Value.Replace(',', '.');
            if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
            {
                list.Add(v);
            }
        }
        return list;
    }

    private static List<PointPair> ToNormalizedDb(List<PointPair> src)
    {
        if (src.Count == 0)
        {
            return [];
        }

        var negatives = src.Count(p => p.Value < 0.0);
        var treatAsDb = negatives > src.Count / 3;

        var outDb = new List<PointPair>(src.Count);
        if (treatAsDb)
        {
            var peak = src.Max(p => p.Value);
            foreach (var p in src)
            {
                var db = Math.Max(PlotFloorDb, p.Value - peak);
                outDb.Add(new PointPair(p.Angle, db));
            }
            return outDb;
        }

        var maxLin = src.Max(p => p.Value);
        if (maxLin <= 0.0)
        {
            maxLin = 1.0;
        }
        foreach (var p in src)
        {
            var lin = Math.Max(1e-12, p.Value / maxLin);
            var db = 20.0 * Math.Log10(lin);
            db = Math.Max(PlotFloorDb, db);
            outDb.Add(new PointPair(p.Angle, db));
        }
        return outDb;
    }

    private void UpdateMetricsGrid()
    {
        gridMetrics.Rows.Clear();

        var mAz = BuildMetrics(_hrpRaw, _hrpDb);
        var mEl = BuildMetrics(_vrpRaw, _vrpDb);

        gridMetrics.Rows.Add("Pontos AZ", mAz.Points.ToString(CultureInfo.InvariantCulture));
        gridMetrics.Rows.Add("Pico AZ [deg]", mAz.PeakAngle.ToString("F2", CultureInfo.InvariantCulture));
        gridMetrics.Rows.Add("HPBW AZ [deg]", mAz.HpbwDegText);
        gridMetrics.Rows.Add("Min AZ [dB]", mAz.MinDb.ToString("F2", CultureInfo.InvariantCulture));
        gridMetrics.Rows.Add("Pontos EL", mEl.Points.ToString(CultureInfo.InvariantCulture));
        gridMetrics.Rows.Add("Pico EL [deg]", mEl.PeakAngle.ToString("F2", CultureInfo.InvariantCulture));
        gridMetrics.Rows.Add("HPBW EL [deg]", mEl.HpbwDegText);
        gridMetrics.Rows.Add("Min EL [dB]", mEl.MinDb.ToString("F2", CultureInfo.InvariantCulture));
    }

    private sealed class Metrics
    {
        public int Points { get; set; }
        public double PeakAngle { get; set; }
        public string HpbwDegText { get; set; } = "-";
        public double MinDb { get; set; }
    }

    private static Metrics BuildMetrics(List<PointPair> raw, List<PointPair> db)
    {
        var m = new Metrics();
        if (raw.Count == 0 || db.Count == 0)
        {
            return m;
        }

        m.Points = raw.Count;
        var peakRaw = raw.Aggregate((a, b) => a.Value >= b.Value ? a : b);
        m.PeakAngle = peakRaw.Angle;
        m.MinDb = db.Min(p => p.Value);
        var hpbw = EstimateHpbw(db);
        m.HpbwDegText = double.IsFinite(hpbw) ? hpbw.ToString("F2", CultureInfo.InvariantCulture) : "-";
        return m;
    }

    private static double EstimateHpbw(List<PointPair> db)
    {
        if (db.Count < 5)
        {
            return double.NaN;
        }
        var peakIdx = 0;
        var peak = db[0].Value;
        for (var i = 1; i < db.Count; i++)
        {
            if (db[i].Value > peak)
            {
                peak = db[i].Value;
                peakIdx = i;
            }
        }

        var target = peak - 3.0;
        var left = InterpCrossing(db, peakIdx, -1, target);
        var right = InterpCrossing(db, peakIdx, +1, target);
        if (!double.IsFinite(left) || !double.IsFinite(right) || right <= left)
        {
            return double.NaN;
        }
        return right - left;
    }

    private static double InterpCrossing(List<PointPair> db, int from, int step, double target)
    {
        for (var i = from + step; i >= 0 && i < db.Count; i += step)
        {
            var a = db[i - step];
            var b = db[i];
            var da = a.Value - target;
            var dbv = b.Value - target;
            if (da == 0.0)
            {
                return a.Angle;
            }
            if (da * dbv <= 0.0)
            {
                var t = (target - a.Value) / (b.Value - a.Value + 1e-12);
                return a.Angle + (b.Angle - a.Angle) * t;
            }
        }
        return double.NaN;
    }

    private void pnlHrp_Paint(object? sender, PaintEventArgs e)
    {
        DrawPolarCut(e.Graphics, pnlHrp.ClientRectangle, _hrpDb);
    }

    private void pnlVrp_Paint(object? sender, PaintEventArgs e)
    {
        DrawPlanarCut(e.Graphics, pnlVrp.ClientRectangle, _vrpDb);
    }

    private void pnlPlot_Resize(object? sender, EventArgs e)
    {
        pnlHrp.Invalidate();
        pnlVrp.Invalidate();
    }

    private static void DrawPolarCut(Graphics g, Rectangle rc, List<PointPair> db)
    {
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.Clear(Color.White);
        using var font = new Font("Segoe UI", 9f);
        using var axisPen = new Pen(Color.FromArgb(210, 210, 210), 1f);
        using var ringPen = new Pen(Color.FromArgb(190, 190, 190), 1f);
        using var curvePen = new Pen(Color.FromArgb(27, 82, 167), 2.2f);

        if (db.Count == 0)
        {
            TextRenderer.DrawText(g, "Sem dados para plotar.", font, rc, Color.DimGray, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
            return;
        }

        var cx = rc.Left + rc.Width / 2f;
        var cy = rc.Top + rc.Height / 2f;
        var radius = Math.Min(rc.Width, rc.Height) * 0.40f;
        var inner = radius * 0.20f;

        for (var tick = -40; tick <= 0; tick += 10)
        {
            var r = MapDbToRadius(tick, inner, radius);
            g.DrawEllipse(ringPen, cx - r, cy - r, 2f * r, 2f * r);
            var label = $"{tick} dB";
            g.DrawString(label, font, Brushes.DimGray, cx + 6f, cy - r - 12f);
        }

        for (var a = -180; a <= 180; a += 30)
        {
            var th = DegToRad(90.0 - a);
            var x = cx + radius * (float)Math.Cos(th);
            var y = cy - radius * (float)Math.Sin(th);
            g.DrawLine(axisPen, cx, cy, x, y);
            var lx = cx + (radius + 16f) * (float)Math.Cos(th);
            var ly = cy - (radius + 16f) * (float)Math.Sin(th);
            g.DrawString(a.ToString(CultureInfo.InvariantCulture), font, Brushes.Gray, lx - 10f, ly - 9f);
        }

        var pts = new List<PointF>(db.Count + 1);
        foreach (var p in db.OrderBy(p => p.Angle))
        {
            var th = DegToRad(90.0 - p.Angle);
            var r = MapDbToRadius(p.Value, inner, radius);
            var x = cx + r * (float)Math.Cos(th);
            var y = cy - r * (float)Math.Sin(th);
            pts.Add(new PointF(x, y));
        }
        if (pts.Count > 1)
        {
            pts.Add(pts[0]);
            g.DrawLines(curvePen, [.. pts]);
        }
    }

    private static void DrawPlanarCut(Graphics g, Rectangle rc, List<PointPair> db)
    {
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.Clear(Color.White);
        using var font = new Font("Segoe UI", 9f);
        using var gridPen = new Pen(Color.FromArgb(215, 215, 215), 1f);
        using var axisPen = new Pen(Color.FromArgb(120, 120, 120), 1.3f);
        using var curvePen = new Pen(Color.FromArgb(16, 122, 83), 2.2f);

        if (db.Count == 0)
        {
            TextRenderer.DrawText(g, "Sem dados para plotar.", font, rc, Color.DimGray, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
            return;
        }

        var plot = Rectangle.FromLTRB(rc.Left + 62, rc.Top + 24, rc.Right - 28, rc.Bottom - 48);
        if (plot.Width < 20 || plot.Height < 20)
        {
            return;
        }

        for (var y = -40; y <= 0; y += 10)
        {
            var yy = MapY(y, plot);
            g.DrawLine(gridPen, plot.Left, yy, plot.Right, yy);
            g.DrawString(y.ToString(CultureInfo.InvariantCulture), font, Brushes.Gray, 8f, yy - 9f);
        }
        for (var x = -90; x <= 90; x += 30)
        {
            var xx = MapX(x, -90.0, 90.0, plot);
            g.DrawLine(gridPen, xx, plot.Top, xx, plot.Bottom);
            g.DrawString(x.ToString(CultureInfo.InvariantCulture), font, Brushes.Gray, xx - 10f, plot.Bottom + 6f);
        }
        g.DrawRectangle(axisPen, plot);

        var pts = db
            .OrderBy(p => p.Angle)
            .Select(p => new PointF(MapX(p.Angle, -90.0, 90.0, plot), MapY(p.Value, plot)))
            .ToArray();
        if (pts.Length > 1)
        {
            g.DrawLines(curvePen, pts);
        }

        g.DrawString("Angulo (deg)", font, Brushes.DimGray, plot.Left + plot.Width / 2f - 34f, plot.Bottom + 24f);
        g.TranslateTransform(18f, plot.Top + plot.Height / 2f + 30f);
        g.RotateTransform(-90f);
        g.DrawString("Nivel (dB)", font, Brushes.DimGray, 0f, 0f);
        g.ResetTransform();
    }

    private static float MapDbToRadius(double db, float inner, float outer)
    {
        var t = (db - PlotFloorDb) / (0.0 - PlotFloorDb);
        t = Math.Clamp(t, 0.0, 1.0);
        return inner + (float)t * (outer - inner);
    }

    private static float MapX(double x, double minX, double maxX, Rectangle plot)
    {
        var t = (x - minX) / (maxX - minX);
        t = Math.Clamp(t, 0.0, 1.0);
        return plot.Left + (float)t * plot.Width;
    }

    private static float MapY(double db, Rectangle plot)
    {
        var t = (db - PlotFloorDb) / (0.0 - PlotFloorDb);
        t = Math.Clamp(t, 0.0, 1.0);
        return plot.Bottom - (float)t * plot.Height;
    }

    private static double DegToRad(double deg) => deg * Math.PI / 180.0;

    private static string SanitizeName(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            sb.Append(char.IsLetterOrDigit(c) || c == '_' || c == '-' ? c : '_');
        }
        var s = sb.ToString().Trim('_');
        return string.IsNullOrWhiteSpace(s) ? "Projeto_VisualStudio" : s;
    }

    private static void WriteProjectJson(string path, string baseName, List<PointPair> hrp, List<PointPair> vrp)
    {
        using var fs = File.Create(path);
        using var jw = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true });
        jw.WriteStartObject();
        jw.WriteString("base_name_var", baseName);

        jw.WritePropertyName("study_h1_angles");
        jw.WriteStartArray();
        foreach (var p in hrp)
        {
            jw.WriteNumberValue(p.Angle);
        }
        jw.WriteEndArray();

        jw.WritePropertyName("study_h1_vals");
        jw.WriteStartArray();
        foreach (var p in hrp)
        {
            jw.WriteNumberValue(p.Value);
        }
        jw.WriteEndArray();

        jw.WritePropertyName("study_v1_angles");
        jw.WriteStartArray();
        foreach (var p in vrp)
        {
            jw.WriteNumberValue(p.Angle);
        }
        jw.WriteEndArray();

        jw.WritePropertyName("study_v1_vals");
        jw.WriteStartArray();
        foreach (var p in vrp)
        {
            jw.WriteNumberValue(p.Value);
        }
        jw.WriteEndArray();

        jw.WriteEndObject();
        jw.Flush();
    }

    private async Task<int> RunProcessAsync(string fileName, IEnumerable<string> args, string workdir)
    {
        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            WorkingDirectory = workdir,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        foreach (var a in args)
        {
            psi.ArgumentList.Add(a);
        }

        using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        proc.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
            {
                Log(e.Data!);
            }
        };
        proc.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
            {
                Log("[stderr] " + e.Data);
            }
        };

        var argsText = string.Join(" ", psi.ArgumentList.Select(QuoteArg));
        Log($"> {Path.GetFileName(fileName)} {argsText}");
        proc.Start();
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        await proc.WaitForExitAsync();
        Log($"Processo finalizado com codigo {proc.ExitCode}.");
        return proc.ExitCode;
    }

    private static string QuoteArg(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return "\"\"";
        }
        return s.Contains(' ') || s.Contains('\t') ? $"\"{s}\"" : s;
    }
}
