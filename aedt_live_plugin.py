"""Drop-in plugin entrypoint for AEDT Live integration.

Integration pattern (minimal changes in your existing app):

1) Add this repo folder to PYTHONPATH (same as your app root).
2) In your main UI file (e.g., deep3.py), after creating your tabview:

    from aedt_live_plugin import register_aedt_live_tab
    register_aedt_live_tab(app=self, tabview=self.tabview)

This keeps changes minimal and avoids refactors.

The tab is implemented in `eftx_aedt_live.ui_tab.AedtLiveTab`.
"""

from __future__ import annotations

from typing import Optional

try:
    import customtkinter as ctk
except Exception as e:  # pragma: no cover
    raise ImportError("customtkinter is required to register AEDT Live UI") from e

from eftx_aedt_live.ui_tab import AedtLiveTab


def register_aedt_live_tab(app, tabview: "ctk.CTkTabview", tab_name: str = "AEDT Live", output_dir: Optional[str] = None):
    """Register the AEDT Live tab into an existing CTkTabview.

    Parameters:
      app: your main application instance (optional; used for bridge callbacks)
      tabview: the CTkTabview instance where tabs are added
      tab_name: display name
      output_dir: directory to export retrieved patterns (defaults to ./aedt_exports)

    Returns:
      The created AedtLiveTab instance.
    """
    tab = tabview.add(tab_name)
    frame = AedtLiveTab(tab, app=app, output_dir=output_dir)
    frame.pack(fill="both", expand=True)
    return frame
