EFTX AEDT LIVE PLUGIN (HFSS post-processing)

Contents
- aedt_live_plugin.py: entrypoint to register a new "AEDT Live" tab in your CTkTabview.
- eftx_aedt_live/: internal package with session, far-field extractor, exports, UI.

How to integrate (minimal change)
1) Copy the folder `eftx_aedt_live_plugin/` contents into your app repo root.
2) In your main CTk app, after you create the CTkTabview instance:

   from aedt_live_plugin import register_aedt_live_tab
   register_aedt_live_tab(app=self, tabview=self.tabs, tab_name="AEDT Live", output_dir=self.output_dir)

3) Run the app. The new tab lets you connect to AEDT 2025.2 and pull VRP/HRP and a 3D grid.

Notes
- This integration avoids refactors but requires that single import+call hook.
- The plugin exports patterns to ./aedt_exports by default.
