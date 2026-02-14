
import flet as ft
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import base64
import io
import numpy as np
import logic

class AntennaPatternView(ft.UserControl):
    def __init__(self, label: str, color: str, page: ft.Page):
        super().__init__()
        self.label = label
        self.plot_color = color
        self.page_ref = page
        
        # State
        self.angles = None # np.array
        self.values_linear = None # np.array
        self.filename = "Nenhum arquivo"
        self.plot_mode = "polar" # polar, planar
        self.table_unit = "linear" # linear, db
        
        # UI Components
        self.img_plot = ft.Image(src_base64="", width=400, height=400, fit="contain", visible=False)
        self.data_table = ft.DataTable(columns=[], rows=[], visible=False)
        self.file_picker = ft.FilePicker(on_result=self.load_file_result)
        self.save_plot_picker = ft.FilePicker(on_result=self.save_plot_result)
        self.save_table_img_picker = ft.FilePicker(on_result=self.save_table_img_result)
        self.save_csv_picker = ft.FilePicker(on_result=self.save_csv_result)
        
        self.lbl_file = ft.Text(f"{self.filename}", italic=True)
        self.lbl_status = ft.Text("Aguardando arquivo...")

    def build(self):
        # Register pickers
        self.page_ref.overlay.extend([
            self.file_picker, 
            self.save_plot_picker, 
            self.save_table_img_picker,
            self.save_csv_picker
        ])
        
        # Controls
        self.btn_load = ft.ElevatedButton(f"Carregar {self.label}", icon=ft.icons.UPLOAD, 
                                          on_click=lambda _: self.file_picker.pick_files())
        
        self.toggle_plot = ft.SegmentedButton(
            segments=[
                ft.Segment(value="polar", label=ft.Text("Polar")),
                ft.Segment(value="planar", label=ft.Text("Planar")),
            ],
            selected={"polar"},
            on_change=self.change_plot_mode
        )
        
        self.toggle_unit = ft.SegmentedButton(
            segments=[
                ft.Segment(value="linear", label=ft.Text("Linear")),
                ft.Segment(value="db", label=ft.Text("dB")),
            ],
            selected={"linear"},
            on_change=self.change_unit
        )
        
        # Plot Section
        plot_col = ft.Column([
            ft.Row([ft.Text("Modo de Plotagem:"), self.toggle_plot], alignment="center"),
            ft.Container(self.img_plot, alignment=ft.alignment.center, border=ft.border.all(1, ft.colors.GREY_800)),
            ft.Row([
                ft.ElevatedButton("Exportar Imagem do Gráfico", icon=ft.icons.IMAGE, 
                                  on_click=lambda _: self.save_plot_picker.save_file(allowed_extensions=["png"], file_name=f"{self.label}_plot.png"))
            ], alignment="center")
        ], spacing=20, visible=False)
        self.plot_container = plot_col
        
        # Table Section
        table_col = ft.Column([
            ft.Row([ft.Text("Unidade:"), self.toggle_unit], alignment="center"),
            ft.Container(
                content=ft.Column([self.data_table], scroll="auto", height=300),
                border=ft.border.all(1, ft.colors.GREY_800),
                padding=10
            ),
            ft.Row([
                ft.ElevatedButton("Exportar Imagem da Tabela", icon=ft.icons.IMAGE,
                                  on_click=lambda _: self.save_table_img_picker.save_file(allowed_extensions=["png"], file_name=f"{self.label}_table.png")),
                ft.ElevatedButton("Exportar CSV", icon=ft.icons.TABLE_VIEW,
                                  on_click=lambda _: self.save_csv_picker.save_file(allowed_extensions=["csv"], file_name=f"{self.label}_data.csv")),
            ], alignment="center")
        ], spacing=20, visible=False)
        self.table_container = table_col

        return ft.Container(
            padding=10,
            content=ft.Column([
                ft.Row([self.btn_load, self.lbl_file], alignment="center"),
                ft.Divider(),
                self.lbl_status,
                self.plot_container,
                ft.Divider(),
                self.table_container
            ], scroll="auto")
        )

    # --- Logic ---

    def load_file_result(self, e: ft.FilePickerResultEvent):
        if e.files:
            path = e.files[0].path
            name = e.files[0].name
            self.lbl_file.value = name
            self.lbl_status.value = "Processando..."
            self.update()
            
            try:
                ang, val = logic.parse_auto(path)
                self.angles = ang
                self.values_linear = logic.normalize_linear(val)
                self.lbl_status.value = f"Carregado: {len(ang)} pontos."
                
                self.plot_container.visible = True
                self.table_container.visible = True
                
                self.refresh_all()
            except Exception as ex:
                self.lbl_status.value = f"Erro: {ex}"
                self.plot_container.visible = False
                self.table_container.visible = False
                self.update()

    def change_plot_mode(self, e):
        if isinstance(e.control.selected, set):
            self.plot_mode = list(e.control.selected)[0]
        self.update_plot()
        
    def change_unit(self, e):
        if isinstance(e.control.selected, set):
            self.table_unit = list(e.control.selected)[0]
        self.update_table()

    def get_current_values(self):
        if self.values_linear is None: return None
        if self.table_unit == "db":
            return logic.linear_to_db(self.values_linear)
        return self.values_linear

    def refresh_all(self):
        self.update_plot()
        self.update_table()

    def update_plot(self):
        if self.angles is None: return
        
        vals = self.values_linear # Plot always linear normalized 0-1 usually, but let's stick to standard E/Emax
        # If user wants dB plot, we might need another toggle. Usually separate for HRP/VRP plots. 
        # Standard: Polar is linear? Or dB? 
        # User requested "Plotagem planar ou polar". Usually E/Emax or dB. 
        # Let's assume the plot follows the "Unidade" toggle? 
        # Or keep plot normalized linear (0..1) as per previous version?
        # Previous version: normalized linear.
        # User said "4 colunas de grau com o respectivo E/Emax ou Db o user escolhe".
        # This applied to the table.
        # For plot, usually we want to see pattern shape.
        # Let's keep plot as E/Emax (linear) for now as originally implemented, 
        # unless specifically requested otherwise for plot.
        
        # NOTE: Updated to use linear for consistence with previous iteration, 
        # but could link to table unit if desired. For now, independent.
        vals_plot = self.values_linear

        fig = matplotlib.figure.Figure(figsize=(5, 5), dpi=100)
        
        if self.plot_mode == "polar":
            ax = fig.add_subplot(111, projection="polar")
            theta = np.deg2rad(self.angles)
            ax.plot(theta, vals_plot, color=self.plot_color, linewidth=2)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_yticks([0.2, 0.5, 0.8, 1.0])
            ax.grid(True, alpha=0.3)
        else:
            ax = fig.add_subplot(111)
            ax.plot(self.angles, vals_plot, color=self.plot_color, linewidth=2)
            ax.set_xlabel("Ângulo (º)")
            ax.set_ylabel("Amplitude (Linear)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(min(self.angles), max(self.angles))

        buf = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        self.img_plot.src_base64 = data
        self.img_plot.visible = True
        self.update()

    def update_table(self):
        if self.angles is None: return
        
        vals = self.get_current_values()
        
        # Create columns
        self.data_table.columns = [
            ft.DataColumn(ft.Text("Ângulo")),
            ft.DataColumn(ft.Text(f"Valor ({self.table_unit})")),
        ]
        
        # Create rows (limit for performance if too many?)
        # Flet DataTable can be slow with thousands of rows. 
        # Let's show first 100 or something? Or paginated?
        # User wants "Tabela de dados". 
        # Let's show all but use a smaller height container.
        
        rows = []
        # Optimization: Don't render thousands of rows in Flet UI if not needed.
        # But for "export", we need full data.
        # Ideally we paginate.
        # For this demo, let's limit to 100 rows preview, but export full.
        
        preview_limit = 50
        for i in range(min(len(self.angles), preview_limit)):
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(f"{self.angles[i]:.1f}")),
                ft.DataCell(ft.Text(f"{vals[i]:.4f}")),
            ]))
            
        if len(self.angles) > preview_limit:
            rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text("...")), 
                ft.DataCell(ft.Text(f"(+ {len(self.angles)-preview_limit} linhas)"))
            ]))
            
        self.data_table.rows = rows
        self.update()

    # --- Export ---

    def save_plot_result(self, e: ft.FilePickerResultEvent):
        if e.path and self.img_plot.src_base64:
            try:
                # Decode base64 back to bytes
                data = base64.b64decode(self.img_plot.src_base64)
                with open(e.path, "wb") as f:
                    f.write(data)
                self.page_ref.show_snack_bar(ft.SnackBar(ft.Text("Gráfico salvo!")))
            except Exception as ex:
                self.page_ref.show_snack_bar(ft.SnackBar(ft.Text(f"Erro: {ex}")))

    def save_table_img_result(self, e: ft.FilePickerResultEvent):
        if e.path and self.angles is not None:
            try:
                vals = self.get_current_values()
                b64_data = logic.render_table_image(self.angles, vals, self.table_unit, self.plot_color)
                data = base64.b64decode(b64_data)
                with open(e.path, "wb") as f:
                    f.write(data)
                self.page_ref.show_snack_bar(ft.SnackBar(ft.Text("Imagem da tabela salva!")))
            except Exception as ex:
                self.page_ref.show_snack_bar(ft.SnackBar(ft.Text(f"Erro: {ex}")))

    def save_csv_result(self, e: ft.FilePickerResultEvent):
        if e.path and self.angles is not None:
            try:
                vals = self.get_current_values()
                import csv
                with open(e.path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Angle", f"Value_{self.table_unit}"])
                    for a, v in zip(self.angles, vals):
                        writer.writerow([a, v])
                self.page_ref.show_snack_bar(ft.SnackBar(ft.Text("CSV salvo!")))
            except Exception as ex:
                self.page_ref.show_snack_bar(ft.SnackBar(ft.Text(f"Erro: {ex}")))


class MobileApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "EFTX Converter Mobile"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 10
        self.page.scroll = "auto"
        
        # Views
        self.hrp_view = AntennaPatternView("HRP", "tab:blue", page)
        self.vrp_view = AntennaPatternView("VRP", "tab:red", page)
        
        # Layout
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(text="HRP (Horizontal)", content=self.hrp_view),
                ft.Tab(text="VRP (Vertical)", content=self.vrp_view),
            ],
            expand=True,
        )
        
        self.page.add(
            ft.Text("EFTX Mobile Converter", size=24, weight="bold"),
            self.tabs
        )

def main(page: ft.Page):
    MobileApp(page)

if __name__ == "__main__":
    ft.app(target=main)
