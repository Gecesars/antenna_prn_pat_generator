
import flet as ft
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import base64
import io
import numpy as np
import logic  # Our ported logic

class MobileApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "EFTX Converter Mobile"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 10
        
        # State
        self.current_angles = None
        self.current_values = None
        self.filename = "Sem arquivo"
        
        # UI Elements
        self.img_plot = ft.Image(src_base64="", width=350, height=350, fit=ft.ImageFit.CONTAIN)
        self.lbl_status = ft.Text("Selecione um arquivo .csv ou .txt")
        self.lbl_file = ft.Text(self.filename, size=20, weight=ft.FontWeight.BOLD)
        
        # File Picker
        self.file_picker = ft.FilePicker(on_result=self.pick_files_result)
        self.page.overlay.append(self.file_picker)
        
        # Build UI
        self.page.add(
            ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Text("EFTX Mobile", size=30, weight=ft.FontWeight.BOLD),
                        self.lbl_file,
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=20
                ),
                ft.Container(
                    content=self.img_plot,
                    alignment=ft.alignment.center,
                    border=ft.border.all(1, ft.colors.GREY_800),
                    border_radius=10,
                    padding=10
                ),
                ft.Row([
                    ft.ElevatedButton("Importar CSV/TXT", icon=ft.icons.UPLOAD_FILE, 
                                     on_click=lambda _: self.file_picker.pick_files(allow_multiple=False)),
                ], alignment=ft.MainAxisAlignment.CENTER),
                
                ft.Divider(),
                
                ft.Text("Visualização"),
                ft.Row([
                    ft.SegmentedButton(
                        segments=[
                            ft.Segment(value="polar", label=ft.Text("Polar")),
                            ft.Segment(value="planar", label=ft.Text("Planar")),
                        ],
                        selected={"polar"},
                        on_change=self.change_plot_mode
                    )
                ], alignment=ft.MainAxisAlignment.CENTER),
                
                self.lbl_status
            ], scroll=ft.ScrollMode.AUTO, expand=True)
        )
        
        self.plot_mode = "polar"

    def pick_files_result(self, e: ft.FilePickerResultEvent):
        if e.files:
            path = e.files[0].path
            name = e.files[0].name
            self.lbl_file.value = name
            self.lbl_status.value = f"Carregando {name}..."
            self.page.update()
            
            try:
                # Use logic to parse
                ang, val = logic.parse_auto(path)
                # Normalize
                self.current_values = logic.normalize_linear(val)
                self.current_angles = ang
                self.update_plot()
                self.lbl_status.value = "Carregado com sucesso."
            except Exception as ex:
                self.lbl_status.value = f"Erro: {ex}"
            
            self.page.update()

    def change_plot_mode(self, e):
        # segmented button returns a set of selected values
        if isinstance(e.control.selected, set):
            self.plot_mode = list(e.control.selected)[0]
        self.update_plot()

    def update_plot(self):
        if self.current_angles is None:
            return
            
        fig = matplotlib.figure.Figure(figsize=(5, 5), dpi=100)
        fig.patch.set_alpha(0) # Transparent bg logic if needed
        
        if self.plot_mode == "polar":
            ax = fig.add_subplot(111, projection="polar")
            theta = np.deg2rad(self.current_angles)
            ax.plot(theta, self.current_values, color="tab:blue")
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_yticks([0.2, 0.5, 0.8, 1.0])
            ax.grid(True, alpha=0.3)
        else:
            ax = fig.add_subplot(111)
            ax.plot(self.current_angles, self.current_values, color="tab:orange")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Normalized")
            ax.grid(True, alpha=0.3)
            
        # Convert to Image
        buf = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        self.img_plot.src_base64 = data
        self.page.update()

def main(page: ft.Page):
    MobileApp(page)

if __name__ == "__main__":
    ft.app(target=main)
