from __future__ import annotations

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PlotPanel(ctk.CTkFrame):
    def __init__(self, master, polar: bool = False, title: str = "", **kwargs):
        super().__init__(master, **kwargs)
        self.figure = Figure(figsize=(7.2, 5.4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="polar" if polar else None)
        self.ax.set_title(title)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def reset_axes(self, polar: bool, title: str):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111, projection="polar" if polar else None)
        self.ax.set_title(title)
        self.canvas.draw_idle()
