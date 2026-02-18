namespace EFTX.PatternStudio;

partial class Form1
{
    /// <summary>
    ///  Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null;

    /// <summary>
    ///  Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
        {
            components.Dispose();
        }
        base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    /// <summary>
    ///  Required method for Designer support - do not modify
    ///  the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent()
    {
        lblCoreExe = new Label();
        txtCoreExe = new TextBox();
        btnBrowseCoreExe = new Button();
        btnBuildCore = new Button();
        lblHrp = new Label();
        txtHrp = new TextBox();
        btnBrowseHrp = new Button();
        lblVrp = new Label();
        txtVrp = new TextBox();
        btnBrowseVrp = new Button();
        lblOutput = new Label();
        txtOutput = new TextBox();
        btnBrowseOutput = new Button();
        btnOpenOutput = new Button();
        lblBaseName = new Label();
        txtBaseName = new TextBox();
        lblNorm = new Label();
        cmbNorm = new ComboBox();
        btnPreview = new Button();
        btnExport = new Button();
        layoutRoot = new TableLayoutPanel();
        grpInput = new GroupBox();
        inputGrid = new TableLayoutPanel();
        splitMain = new SplitContainer();
        tabsPlots = new TabControl();
        tabHrp = new TabPage();
        pnlHrp = new Panel();
        tabVrp = new TabPage();
        pnlVrp = new Panel();
        rightLayout = new TableLayoutPanel();
        grpMetrics = new GroupBox();
        gridMetrics = new DataGridView();
        colMetric = new DataGridViewTextBoxColumn();
        colValue = new DataGridViewTextBoxColumn();
        grpLog = new GroupBox();
        txtLog = new TextBox();
        layoutRoot.SuspendLayout();
        grpInput.SuspendLayout();
        inputGrid.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)splitMain).BeginInit();
        splitMain.Panel1.SuspendLayout();
        splitMain.Panel2.SuspendLayout();
        splitMain.SuspendLayout();
        tabsPlots.SuspendLayout();
        tabHrp.SuspendLayout();
        tabVrp.SuspendLayout();
        rightLayout.SuspendLayout();
        grpMetrics.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)gridMetrics).BeginInit();
        grpLog.SuspendLayout();
        SuspendLayout();
        // 
        // lblCoreExe
        // 
        lblCoreExe.Anchor = AnchorStyles.Left;
        lblCoreExe.AutoSize = true;
        lblCoreExe.Location = new Point(3, 8);
        lblCoreExe.Name = "lblCoreExe";
        lblCoreExe.Size = new Size(100, 20);
        lblCoreExe.TabIndex = 0;
        lblCoreExe.Text = "Core C++ EXE:";
        // 
        // txtCoreExe
        // 
        txtCoreExe.Dock = DockStyle.Fill;
        txtCoreExe.Location = new Point(109, 3);
        txtCoreExe.Name = "txtCoreExe";
        txtCoreExe.Size = new Size(590, 27);
        txtCoreExe.TabIndex = 1;
        // 
        // btnBrowseCoreExe
        // 
        btnBrowseCoreExe.Dock = DockStyle.Fill;
        btnBrowseCoreExe.Location = new Point(705, 3);
        btnBrowseCoreExe.Name = "btnBrowseCoreExe";
        btnBrowseCoreExe.Size = new Size(92, 31);
        btnBrowseCoreExe.TabIndex = 2;
        btnBrowseCoreExe.Text = "Procurar";
        btnBrowseCoreExe.UseVisualStyleBackColor = true;
        btnBrowseCoreExe.Click += btnBrowseCoreExe_Click;
        // 
        // btnBuildCore
        // 
        btnBuildCore.Dock = DockStyle.Fill;
        btnBuildCore.Location = new Point(803, 3);
        btnBuildCore.Name = "btnBuildCore";
        btnBuildCore.Size = new Size(133, 31);
        btnBuildCore.TabIndex = 3;
        btnBuildCore.Text = "Build Core";
        btnBuildCore.UseVisualStyleBackColor = true;
        btnBuildCore.Click += btnBuildCore_Click;
        // 
        // lblHrp
        // 
        lblHrp.Anchor = AnchorStyles.Left;
        lblHrp.AutoSize = true;
        lblHrp.Location = new Point(3, 45);
        lblHrp.Name = "lblHrp";
        lblHrp.Size = new Size(78, 20);
        lblHrp.TabIndex = 4;
        lblHrp.Text = "Arquivo AZ:";
        // 
        // txtHrp
        // 
        txtHrp.Dock = DockStyle.Fill;
        txtHrp.Location = new Point(109, 40);
        txtHrp.Name = "txtHrp";
        txtHrp.Size = new Size(590, 27);
        txtHrp.TabIndex = 5;
        // 
        // btnBrowseHrp
        // 
        btnBrowseHrp.Dock = DockStyle.Fill;
        btnBrowseHrp.Location = new Point(705, 40);
        btnBrowseHrp.Name = "btnBrowseHrp";
        btnBrowseHrp.Size = new Size(92, 31);
        btnBrowseHrp.TabIndex = 6;
        btnBrowseHrp.Text = "Procurar";
        btnBrowseHrp.UseVisualStyleBackColor = true;
        btnBrowseHrp.Click += btnBrowseHrp_Click;
        // 
        // lblVrp
        // 
        lblVrp.Anchor = AnchorStyles.Left;
        lblVrp.AutoSize = true;
        lblVrp.Location = new Point(3, 82);
        lblVrp.Name = "lblVrp";
        lblVrp.Size = new Size(80, 20);
        lblVrp.TabIndex = 7;
        lblVrp.Text = "Arquivo EL:";
        // 
        // txtVrp
        // 
        txtVrp.Dock = DockStyle.Fill;
        txtVrp.Location = new Point(109, 77);
        txtVrp.Name = "txtVrp";
        txtVrp.Size = new Size(590, 27);
        txtVrp.TabIndex = 8;
        // 
        // btnBrowseVrp
        // 
        btnBrowseVrp.Dock = DockStyle.Fill;
        btnBrowseVrp.Location = new Point(705, 77);
        btnBrowseVrp.Name = "btnBrowseVrp";
        btnBrowseVrp.Size = new Size(92, 31);
        btnBrowseVrp.TabIndex = 9;
        btnBrowseVrp.Text = "Procurar";
        btnBrowseVrp.UseVisualStyleBackColor = true;
        btnBrowseVrp.Click += btnBrowseVrp_Click;
        // 
        // lblOutput
        // 
        lblOutput.Anchor = AnchorStyles.Left;
        lblOutput.AutoSize = true;
        lblOutput.Location = new Point(3, 119);
        lblOutput.Name = "lblOutput";
        lblOutput.Size = new Size(100, 20);
        lblOutput.TabIndex = 10;
        lblOutput.Text = "Pasta de saida:";
        // 
        // txtOutput
        // 
        txtOutput.Dock = DockStyle.Fill;
        txtOutput.Location = new Point(109, 114);
        txtOutput.Name = "txtOutput";
        txtOutput.Size = new Size(590, 27);
        txtOutput.TabIndex = 11;
        // 
        // btnBrowseOutput
        // 
        btnBrowseOutput.Dock = DockStyle.Fill;
        btnBrowseOutput.Location = new Point(705, 114);
        btnBrowseOutput.Name = "btnBrowseOutput";
        btnBrowseOutput.Size = new Size(92, 31);
        btnBrowseOutput.TabIndex = 12;
        btnBrowseOutput.Text = "Procurar";
        btnBrowseOutput.UseVisualStyleBackColor = true;
        btnBrowseOutput.Click += btnBrowseOutput_Click;
        // 
        // btnOpenOutput
        // 
        btnOpenOutput.Dock = DockStyle.Fill;
        btnOpenOutput.Location = new Point(803, 114);
        btnOpenOutput.Name = "btnOpenOutput";
        btnOpenOutput.Size = new Size(133, 31);
        btnOpenOutput.TabIndex = 13;
        btnOpenOutput.Text = "Abrir Pasta";
        btnOpenOutput.UseVisualStyleBackColor = true;
        btnOpenOutput.Click += btnOpenOutput_Click;
        // 
        // lblBaseName
        // 
        lblBaseName.Anchor = AnchorStyles.Left;
        lblBaseName.AutoSize = true;
        lblBaseName.Location = new Point(3, 155);
        lblBaseName.Name = "lblBaseName";
        lblBaseName.Size = new Size(95, 20);
        lblBaseName.TabIndex = 14;
        lblBaseName.Text = "Nome base:";
        // 
        // txtBaseName
        // 
        txtBaseName.Dock = DockStyle.Fill;
        txtBaseName.Location = new Point(109, 151);
        txtBaseName.Name = "txtBaseName";
        txtBaseName.Size = new Size(590, 27);
        txtBaseName.TabIndex = 15;
        txtBaseName.Text = "Projeto_VisualStudio";
        // 
        // lblNorm
        // 
        lblNorm.Anchor = AnchorStyles.Left;
        lblNorm.AutoSize = true;
        lblNorm.Location = new Point(705, 155);
        lblNorm.Name = "lblNorm";
        lblNorm.Size = new Size(56, 20);
        lblNorm.TabIndex = 16;
        lblNorm.Text = "Norma:";
        // 
        // cmbNorm
        // 
        cmbNorm.Dock = DockStyle.Fill;
        cmbNorm.DropDownStyle = ComboBoxStyle.DropDownList;
        cmbNorm.FormattingEnabled = true;
        cmbNorm.Items.AddRange(new object[] { "none", "max", "rms" });
        cmbNorm.Location = new Point(803, 151);
        cmbNorm.Name = "cmbNorm";
        cmbNorm.Size = new Size(133, 28);
        cmbNorm.TabIndex = 17;
        // 
        // btnPreview
        // 
        btnPreview.Dock = DockStyle.Fill;
        btnPreview.Font = new Font("Segoe UI", 9F, FontStyle.Bold);
        btnPreview.Location = new Point(705, 188);
        btnPreview.Name = "btnPreview";
        btnPreview.Size = new Size(92, 34);
        btnPreview.TabIndex = 18;
        btnPreview.Text = "Preview";
        btnPreview.UseVisualStyleBackColor = true;
        btnPreview.Click += btnPreview_Click;
        // 
        // btnExport
        // 
        btnExport.Dock = DockStyle.Fill;
        btnExport.Font = new Font("Segoe UI", 9F, FontStyle.Bold);
        btnExport.Location = new Point(803, 188);
        btnExport.Name = "btnExport";
        btnExport.Size = new Size(133, 34);
        btnExport.TabIndex = 19;
        btnExport.Text = "Exportar Tudo";
        btnExport.UseVisualStyleBackColor = true;
        btnExport.Click += btnExport_Click;
        // 
        // layoutRoot
        // 
        layoutRoot.ColumnCount = 1;
        layoutRoot.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        layoutRoot.Controls.Add(grpInput, 0, 0);
        layoutRoot.Controls.Add(splitMain, 0, 1);
        layoutRoot.Dock = DockStyle.Fill;
        layoutRoot.Location = new Point(0, 0);
        layoutRoot.Name = "layoutRoot";
        layoutRoot.RowCount = 2;
        layoutRoot.RowStyles.Add(new RowStyle(SizeType.Absolute, 262F));
        layoutRoot.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        layoutRoot.Size = new Size(1482, 853);
        layoutRoot.TabIndex = 0;
        // 
        // grpInput
        // 
        grpInput.Controls.Add(inputGrid);
        grpInput.Dock = DockStyle.Fill;
        grpInput.Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold);
        grpInput.Location = new Point(12, 12);
        grpInput.Margin = new Padding(12);
        grpInput.Name = "grpInput";
        grpInput.Size = new Size(1458, 238);
        grpInput.TabIndex = 0;
        grpInput.TabStop = false;
        grpInput.Text = "Entrada e Operacao";
        // 
        // inputGrid
        // 
        inputGrid.ColumnCount = 4;
        inputGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 106F));
        inputGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        inputGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 98F));
        inputGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 139F));
        inputGrid.Controls.Add(lblCoreExe, 0, 0);
        inputGrid.Controls.Add(txtCoreExe, 1, 0);
        inputGrid.Controls.Add(btnBrowseCoreExe, 2, 0);
        inputGrid.Controls.Add(btnBuildCore, 3, 0);
        inputGrid.Controls.Add(lblHrp, 0, 1);
        inputGrid.Controls.Add(txtHrp, 1, 1);
        inputGrid.Controls.Add(btnBrowseHrp, 2, 1);
        inputGrid.Controls.Add(lblVrp, 0, 2);
        inputGrid.Controls.Add(txtVrp, 1, 2);
        inputGrid.Controls.Add(btnBrowseVrp, 2, 2);
        inputGrid.Controls.Add(lblOutput, 0, 3);
        inputGrid.Controls.Add(txtOutput, 1, 3);
        inputGrid.Controls.Add(btnBrowseOutput, 2, 3);
        inputGrid.Controls.Add(btnOpenOutput, 3, 3);
        inputGrid.Controls.Add(lblBaseName, 0, 4);
        inputGrid.Controls.Add(txtBaseName, 1, 4);
        inputGrid.Controls.Add(lblNorm, 2, 4);
        inputGrid.Controls.Add(cmbNorm, 3, 4);
        inputGrid.Controls.Add(btnPreview, 2, 5);
        inputGrid.Controls.Add(btnExport, 3, 5);
        inputGrid.Dock = DockStyle.Fill;
        inputGrid.Location = new Point(3, 23);
        inputGrid.Name = "inputGrid";
        inputGrid.RowCount = 6;
        inputGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 37F));
        inputGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 37F));
        inputGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 37F));
        inputGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 37F));
        inputGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 37F));
        inputGrid.RowStyles.Add(new RowStyle(SizeType.Absolute, 40F));
        inputGrid.Size = new Size(1452, 212);
        inputGrid.TabIndex = 0;
        // 
        // splitMain
        // 
        splitMain.Dock = DockStyle.Fill;
        splitMain.Location = new Point(12, 274);
        splitMain.Margin = new Padding(12);
        splitMain.Name = "splitMain";
        // 
        // splitMain.Panel1
        // 
        splitMain.Panel1.Controls.Add(tabsPlots);
        // 
        // splitMain.Panel2
        // 
        splitMain.Panel2.Controls.Add(rightLayout);
        splitMain.Size = new Size(1458, 567);
        splitMain.SplitterDistance = 1048;
        splitMain.TabIndex = 1;
        // 
        // tabsPlots
        // 
        tabsPlots.Controls.Add(tabHrp);
        tabsPlots.Controls.Add(tabVrp);
        tabsPlots.Dock = DockStyle.Fill;
        tabsPlots.Location = new Point(0, 0);
        tabsPlots.Name = "tabsPlots";
        tabsPlots.SelectedIndex = 0;
        tabsPlots.Size = new Size(1048, 567);
        tabsPlots.TabIndex = 0;
        // 
        // tabHrp
        // 
        tabHrp.Controls.Add(pnlHrp);
        tabHrp.Location = new Point(4, 29);
        tabHrp.Name = "tabHrp";
        tabHrp.Padding = new Padding(3);
        tabHrp.Size = new Size(1040, 534);
        tabHrp.TabIndex = 0;
        tabHrp.Text = "Azimute (Polar)";
        tabHrp.UseVisualStyleBackColor = true;
        // 
        // pnlHrp
        // 
        pnlHrp.BackColor = Color.White;
        pnlHrp.Dock = DockStyle.Fill;
        pnlHrp.Location = new Point(3, 3);
        pnlHrp.Name = "pnlHrp";
        pnlHrp.Size = new Size(1034, 528);
        pnlHrp.TabIndex = 0;
        pnlHrp.Paint += pnlHrp_Paint;
        pnlHrp.Resize += pnlPlot_Resize;
        // 
        // tabVrp
        // 
        tabVrp.Controls.Add(pnlVrp);
        tabVrp.Location = new Point(4, 29);
        tabVrp.Name = "tabVrp";
        tabVrp.Padding = new Padding(3);
        tabVrp.Size = new Size(1040, 534);
        tabVrp.TabIndex = 1;
        tabVrp.Text = "Elevacao (Planar)";
        tabVrp.UseVisualStyleBackColor = true;
        // 
        // pnlVrp
        // 
        pnlVrp.BackColor = Color.White;
        pnlVrp.Dock = DockStyle.Fill;
        pnlVrp.Location = new Point(3, 3);
        pnlVrp.Name = "pnlVrp";
        pnlVrp.Size = new Size(1034, 528);
        pnlVrp.TabIndex = 0;
        pnlVrp.Paint += pnlVrp_Paint;
        pnlVrp.Resize += pnlPlot_Resize;
        // 
        // rightLayout
        // 
        rightLayout.ColumnCount = 1;
        rightLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
        rightLayout.Controls.Add(grpMetrics, 0, 0);
        rightLayout.Controls.Add(grpLog, 0, 1);
        rightLayout.Dock = DockStyle.Fill;
        rightLayout.Location = new Point(0, 0);
        rightLayout.Name = "rightLayout";
        rightLayout.RowCount = 2;
        rightLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 232F));
        rightLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
        rightLayout.Size = new Size(406, 567);
        rightLayout.TabIndex = 0;
        // 
        // grpMetrics
        // 
        grpMetrics.Controls.Add(gridMetrics);
        grpMetrics.Dock = DockStyle.Fill;
        grpMetrics.Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold);
        grpMetrics.Location = new Point(8, 8);
        grpMetrics.Margin = new Padding(8);
        grpMetrics.Name = "grpMetrics";
        grpMetrics.Size = new Size(390, 216);
        grpMetrics.TabIndex = 0;
        grpMetrics.TabStop = false;
        grpMetrics.Text = "Metricas";
        // 
        // gridMetrics
        // 
        gridMetrics.AllowUserToAddRows = false;
        gridMetrics.AllowUserToDeleteRows = false;
        gridMetrics.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
        gridMetrics.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
        gridMetrics.Columns.AddRange(new DataGridViewColumn[] { colMetric, colValue });
        gridMetrics.Dock = DockStyle.Fill;
        gridMetrics.Location = new Point(3, 23);
        gridMetrics.Name = "gridMetrics";
        gridMetrics.ReadOnly = true;
        gridMetrics.RowHeadersVisible = false;
        gridMetrics.RowHeadersWidth = 51;
        gridMetrics.Size = new Size(384, 190);
        gridMetrics.TabIndex = 0;
        // 
        // colMetric
        // 
        colMetric.HeaderText = "Metrica";
        colMetric.MinimumWidth = 6;
        colMetric.Name = "colMetric";
        colMetric.ReadOnly = true;
        // 
        // colValue
        // 
        colValue.HeaderText = "Valor";
        colValue.MinimumWidth = 6;
        colValue.Name = "colValue";
        colValue.ReadOnly = true;
        // 
        // grpLog
        // 
        grpLog.Controls.Add(txtLog);
        grpLog.Dock = DockStyle.Fill;
        grpLog.Font = new Font("Segoe UI Semibold", 9F, FontStyle.Bold);
        grpLog.Location = new Point(8, 240);
        grpLog.Margin = new Padding(8);
        grpLog.Name = "grpLog";
        grpLog.Size = new Size(390, 319);
        grpLog.TabIndex = 1;
        grpLog.TabStop = false;
        grpLog.Text = "Log";
        // 
        // txtLog
        // 
        txtLog.Dock = DockStyle.Fill;
        txtLog.Font = new Font("Consolas", 9F);
        txtLog.Location = new Point(3, 23);
        txtLog.Multiline = true;
        txtLog.Name = "txtLog";
        txtLog.ReadOnly = true;
        txtLog.ScrollBars = ScrollBars.Vertical;
        txtLog.Size = new Size(384, 293);
        txtLog.TabIndex = 0;
        // 
        // Form1
        // 
        AutoScaleDimensions = new SizeF(8F, 20F);
        AutoScaleMode = AutoScaleMode.Font;
        ClientSize = new Size(1482, 853);
        Controls.Add(layoutRoot);
        MinimumSize = new Size(1200, 760);
        Name = "Form1";
        StartPosition = FormStartPosition.CenterScreen;
        Text = "EFTX Pattern Studio (Visual Studio)";
        layoutRoot.ResumeLayout(false);
        grpInput.ResumeLayout(false);
        inputGrid.ResumeLayout(false);
        inputGrid.PerformLayout();
        splitMain.Panel1.ResumeLayout(false);
        splitMain.Panel2.ResumeLayout(false);
        ((System.ComponentModel.ISupportInitialize)splitMain).EndInit();
        splitMain.ResumeLayout(false);
        tabsPlots.ResumeLayout(false);
        tabHrp.ResumeLayout(false);
        tabVrp.ResumeLayout(false);
        rightLayout.ResumeLayout(false);
        grpMetrics.ResumeLayout(false);
        ((System.ComponentModel.ISupportInitialize)gridMetrics).EndInit();
        grpLog.ResumeLayout(false);
        grpLog.PerformLayout();
        ResumeLayout(false);
    }

    #endregion

    private Label lblCoreExe;
    private TextBox txtCoreExe;
    private Button btnBrowseCoreExe;
    private Button btnBuildCore;
    private Label lblHrp;
    private TextBox txtHrp;
    private Button btnBrowseHrp;
    private Label lblVrp;
    private TextBox txtVrp;
    private Button btnBrowseVrp;
    private Label lblOutput;
    private TextBox txtOutput;
    private Button btnBrowseOutput;
    private Button btnOpenOutput;
    private Label lblBaseName;
    private TextBox txtBaseName;
    private Label lblNorm;
    private ComboBox cmbNorm;
    private Button btnPreview;
    private Button btnExport;
    private TableLayoutPanel layoutRoot;
    private GroupBox grpInput;
    private TableLayoutPanel inputGrid;
    private SplitContainer splitMain;
    private TabControl tabsPlots;
    private TabPage tabHrp;
    private Panel pnlHrp;
    private TabPage tabVrp;
    private Panel pnlVrp;
    private TableLayoutPanel rightLayout;
    private GroupBox grpMetrics;
    private DataGridView gridMetrics;
    private DataGridViewTextBoxColumn colMetric;
    private DataGridViewTextBoxColumn colValue;
    private GroupBox grpLog;
    private TextBox txtLog;
}
