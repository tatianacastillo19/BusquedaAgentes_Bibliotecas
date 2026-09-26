"""
Interfaz Gráfica de Usuario (GUI) interactiva con CustomTkinter y Matplotlib.
Incluye:
- Dropdown con Búsqueda predictiva/predictive search en tiempo real por nombre y localidad de biblioteca.
- Navegación interactiva completa del mapa (Arrastrar/Mover con mouse, Zoom con rueda y botones).
- Cuadro destacado del Algoritmo Ganador con la Menor Distancia Recorrida.
- Diferenciación teórica entre Distancia Real (A*, UCS) y Distancia Estimada (Voraz/Greedy).
"""
import os
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from config.bibliotecas import BIBLIOTECAS
from core.map_loader import load_bogota_graph, get_nearest_node, get_edge_cost, BogotaRoadMap
from agents.bfs_agent import bfs_search
from agents.ucs_agent import ucs_search
from agents.greedy_agent import greedy_search
from agents.astar_agent import astar_search

# Configuración visual de CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SearchableLibrarySelector(ctk.CTkFrame):
    """
    Selector interactivo con barra de búsqueda en tiempo real y menú desplegable.
    Permite filtrar las bibliotecas escribiendo cualquier parte de su nombre o localidad (ej. 'Tunal', 'Tintal', 'Suba').
    """
    def __init__(self, master, libraries_dict: Dict[str, Any], default_val: str, on_change=None, width: int = 300, **kwargs):
        super().__init__(master, fg_color="transparent", width=width, **kwargs)
        self.libraries_dict = libraries_dict
        self.all_names = list(libraries_dict.keys())
        self.on_change = on_change
        self.current_value = default_val if default_val in self.all_names else self.all_names[0]

        # Barra superior: Entrada de texto + Botón de despliegue
        self.search_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.search_bar.pack(fill="x")

        self.entry_var = tk.StringVar(value=self.current_value)
        self.entry = ctk.CTkEntry(
            self.search_bar,
            textvariable=self.entry_var,
            placeholder_text="🔍 Escribe para buscar biblioteca...",
            height=34,
            font=ctk.CTkFont(size=11)
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_toggle = ctk.CTkButton(
            self.search_bar,
            text="▼",
            width=32,
            height=34,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#212a36",
            hover_color="#303e50",
            command=self._toggle_suggestions
        )
        self.btn_toggle.pack(side="right")

        # Menú desplegable de sugerencias filtradas
        self.suggestions_frame = ctk.CTkScrollableFrame(
            self,
            height=135,
            fg_color="#161b22",
            border_width=1,
            border_color="#30363d",
            corner_radius=6
        )
        self.is_open = False

        # Eventos del campo de búsqueda
        self.entry.bind("<KeyRelease>", self._on_key_release)
        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<Return>", self._on_enter_pressed)

    def _on_key_release(self, event=None):
        """Filtra sugerencias dinámicamente según lo que escribe el usuario."""
        if event and event.keysym in ("Return", "Up", "Down", "Escape"):
            return
        query = self.entry_var.get().strip().lower()
        self._populate_suggestions(query)
        if not self.is_open:
            self._show_suggestions()

    def _on_focus_in(self, event=None):
        """Muestra las opciones al hacer clic en la caja de texto."""
        query = self.entry_var.get().strip().lower()
        self._populate_suggestions(query)
        if not self.is_open:
            self._show_suggestions()

    def _on_enter_pressed(self, event=None):
        """Selecciona el primer resultado coincidente al presionar Enter."""
        query = self.entry_var.get().strip().lower()
        matches = self._get_matches(query)
        if matches:
            self._select_item(matches[0])

    def _toggle_suggestions(self):
        """Abre o cierra el menú de bibliotecas."""
        if self.is_open:
            self._hide_suggestions()
        else:
            self._populate_suggestions("")
            self._show_suggestions()

    def _show_suggestions(self):
        if not self.is_open:
            self.suggestions_frame.pack(fill="x", pady=(4, 2))
            self.btn_toggle.configure(text="▲")
            self.is_open = True

    def _hide_suggestions(self):
        if self.is_open:
            self.suggestions_frame.pack_forget()
            self.btn_toggle.configure(text="▼")
            self.is_open = False

    def _get_matches(self, query: str) -> List[str]:
        if not query:
            return self.all_names
        matches = []
        for name in self.all_names:
            loc = self.libraries_dict[name].get("localidad", "").lower()
            if query in name.lower() or query in loc:
                matches.append(name)
        return matches

    def _populate_suggestions(self, query: str):
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        matches = self._get_matches(query)
        if not matches:
            lbl = ctk.CTkLabel(
                self.suggestions_frame,
                text="❌ No hay coincidencias",
                font=ctk.CTkFont(size=10),
                text_color="#8b949e"
            )
            lbl.pack(padx=8, pady=6)
            return

        for name in matches:
            loc = self.libraries_dict[name].get("localidad", "")
            btn = ctk.CTkButton(
                self.suggestions_frame,
                text=f"{name}  [{loc}]",
                anchor="w",
                height=28,
                font=ctk.CTkFont(size=10),
                fg_color="transparent",
                hover_color="#212a36",
                text_color="#f0f6fc",
                command=lambda n=name: self._select_item(n)
            )
            btn.pack(fill="x", padx=2, pady=1)

    def _select_item(self, name: str):
        """Aplica la selección elegida por el usuario."""
        self.current_value = name
        self.entry_var.set(name)
        self._hide_suggestions()
        if self.on_change:
            self.on_change(name)

    def get(self) -> str:
        """Retorna el nombre válido de la biblioteca seleccionada."""
        typed = self.entry_var.get().strip().lower()
        for name in self.all_names:
            if typed == name.lower():
                return name
        # Si el usuario escribió una palabra clave que coincide con una sola biblioteca
        matches = self._get_matches(typed)
        if matches:
            return matches[0]
        return self.current_value

    def set(self, value: str):
        if value in self.all_names:
            self.current_value = value
            self.entry_var.set(value)
            if self.on_change:
                self.on_change(value)


class MapPanZoomController:
    """
    Controlador de eventos para mover (Pan) y hacer Zoom interactivo en el canvas de Matplotlib.
    """
    def __init__(self, ax, canvas, on_reset=None):
        self.ax = ax
        self.canvas = canvas
        self.on_reset = on_reset
        self.press = None

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.dblclick:
            if self.on_reset:
                self.on_reset()
            return
        if event.button == 1:  # Clic izquierdo para arrastrar
            self.press = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_release(self, event):
        self.press = None
        self.canvas.draw_idle()

    def _on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        xpress, ypress, xlim, ylim = self.press
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Zoom centrado en la posición del cursor del ratón."""
        if event.inaxes != self.ax:
            return
        base_scale = 1.3
        if event.button == "up":
            scale_factor = 1.0 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        self.ax.set_xlim([xdata - new_width * (1.0 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1.0 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()

    def zoom_by_factor(self, factor: float):
        """Zoom manual mediante botones de la interfaz."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2.0
        cy = (ylim[0] + ylim[1]) / 2.0
        nw = (xlim[1] - xlim[0]) * factor
        nh = (ylim[1] - ylim[0]) * factor
        self.ax.set_xlim(cx - nw / 2.0, cx + nw / 2.0)
        self.ax.set_ylim(cy - nh / 2.0, cy + nh / 2.0)
        self.canvas.draw_idle()


class BibloRedApp(ctk.CTk):
    """
    Ventana principal de la aplicación BibloRed - Comparador de Rutas y Menor Distancia Recorrida.
    """

    def __init__(self):
        super().__init__()

        self.title("BibloRed Bogotá - Búsqueda de Rutas con Menor Distancia Recorrida")
        self.geometry("1340x880")
        self.minsize(1100, 740)

        # Estado interno
        self.road_map: Optional[BogotaRoadMap] = None
        self.is_loading_graph = False
        self.library_names = list(BIBLIOTECAS.keys())
        self.last_results: List[Dict[str, Any]] = []
        self.table_row_widgets: List[ctk.CTkBaseClass] = []

        # Configuración del Grid principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Construir paneles
        self._build_sidebar()
        self._build_main_panel()

        # Iniciar carga asíncrona del mapa vial
        self._async_load_map()

    def _build_sidebar(self):
        """Construye la barra lateral izquierda con controles y búsqueda predictiva."""
        self.sidebar_frame = ctk.CTkFrame(self, width=340, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)

        # Logo / Título
        title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="📚 BibloRed Bogotá",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(padx=20, pady=(16, 3), anchor="w")

        subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Optimizador de Distancia Recorrida en Red Vial",
            font=ctk.CTkFont(size=12),
            text_color="#a0a0a0"
        )
        subtitle_label.pack(padx=20, pady=(0, 10), anchor="w")

        # Separador
        separator1 = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="#333333")
        separator1.pack(fill="x", padx=15, pady=3)

        # Selector Origen (Buscador predictivo)
        ctk.CTkLabel(
            self.sidebar_frame,
            text="📍 Biblioteca de Origen (Buscar):",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(padx=20, pady=(8, 2), anchor="w")

        self.combo_origen = SearchableLibrarySelector(
            self.sidebar_frame,
            libraries_dict=BIBLIOTECAS,
            default_val=self.library_names[0],
            on_change=self._on_selection_change
        )
        self.combo_origen.pack(padx=20, pady=(0, 6), fill="x")

        # Selector Destino (Buscador predictivo)
        ctk.CTkLabel(
            self.sidebar_frame,
            text="🏁 Biblioteca de Destino (Buscar):",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(padx=20, pady=(4, 2), anchor="w")

        self.combo_destino = SearchableLibrarySelector(
            self.sidebar_frame,
            libraries_dict=BIBLIOTECAS,
            default_val=self.library_names[3] if len(self.library_names) > 3 else self.library_names[1],
            on_change=self._on_selection_change
        )
        self.combo_destino.pack(padx=20, pady=(0, 8), fill="x")

        # Selector de Algoritmo
        ctk.CTkLabel(
            self.sidebar_frame,
            text="⚙️ Modelo / Algoritmo a Ejecutar:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(padx=20, pady=(4, 2), anchor="w")

        self.algo_options = [
            "Comparar Todos (4 Algoritmos)",
            "4. Búsqueda A* (Distancia Real Óptima)",
            "2. Costo Uniforme (UCS - Distancia Real)",
            "3. Búsqueda Voraz (Greedy - Solo Estimada)",
            "1. Búsqueda en Anchura (BFS - Saltos)"
        ]
        self.combo_algo = ctk.CTkComboBox(
            self.sidebar_frame,
            values=self.algo_options,
            width=300,
            height=32
        )
        self.combo_algo.set(self.algo_options[0])
        self.combo_algo.pack(padx=20, pady=(0, 10))

        # Botón de Búsqueda
        self.btn_search = ctk.CTkButton(
            self.sidebar_frame,
            text="🚀 Calcular Menor Ruta",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._on_search_clicked
        )
        self.btn_search.pack(padx=20, pady=(6, 6), fill="x")

        # Barra de progreso y estado
        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame, mode="indeterminate", width=300)
        self.lbl_status = ctk.CTkLabel(
            self.sidebar_frame,
            text="Iniciando...",
            font=ctk.CTkFont(size=12),
            text_color="#ffd166"
        )
        self.lbl_status.pack(padx=20, pady=(3, 3), anchor="w")

        # Tarjeta explicativa teórica fija en la barra lateral
        self.theory_card = ctk.CTkFrame(self.sidebar_frame, corner_radius=6, fg_color="#18202a", border_width=1, border_color="#2c3e50")
        self.theory_card.pack(fill="x", padx=15, pady=(6, 6))

        ctk.CTkLabel(
            self.theory_card,
            text="💡 Fundamento Teórico:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#58a6ff"
        ).pack(padx=10, pady=(6, 2), anchor="w")

        theory_text = (
            "• A* & UCS: Evalúan la distancia real g(n) de las calles, garantizando la ruta más corta.\n\n"
            "• Voraz (Greedy): NO evalúa distancia real; solo estima en línea recta h(n) al destino, resultando en rutas más largas.\n\n"
            "• BFS: Solo cuenta número de giros/aristas."
        )
        ctk.CTkLabel(
            self.theory_card,
            text=theory_text,
            font=ctk.CTkFont(size=10),
            text_color="#c0c8d0",
            justify="left",
            wraplength=285
        ).pack(padx=10, pady=(0, 6), anchor="w")

        # Info footer
        self.lbl_graph_info = ctk.CTkLabel(
            self.sidebar_frame,
            text="Red Vial: Cargando...",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.lbl_graph_info.pack(side="bottom", padx=20, pady=8, anchor="w")

    def _build_main_panel(self):
        """Construye el panel central con el mapa interactivo y el cuadro de menor distancia."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_frame.grid_rowconfigure(0, weight=6)
        self.main_frame.grid_rowconfigure(1, weight=4)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- Subpanel 1: Visualización del Mapa Interactivo (Matplotlib) ---
        self.map_container = ctk.CTkFrame(self.main_frame, corner_radius=8, fg_color="#14181d")
        self.map_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=(8, 4))
        self.map_container.grid_rowconfigure(1, weight=1)
        self.map_container.grid_columnconfigure(0, weight=1)

        # Barra de herramientas superior del mapa
        self.map_toolbar = ctk.CTkFrame(self.map_container, height=36, fg_color="#1c2430", corner_radius=6)
        self.map_toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))

        ctk.CTkLabel(
            self.map_toolbar,
            text="🗺️ Controles: [Arrastrar con Clic] para Mover | [Rueda del Ratón] para Zoom",
            font=ctk.CTkFont(size=11),
            text_color="#8b949e"
        ).pack(side="left", padx=12, pady=4)

        self.btn_reset_zoom = ctk.CTkButton(
            self.map_toolbar,
            text="🔄 Reencuadrar",
            width=100,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#238636",
            hover_color="#2ea043",
            command=self._reset_map_view
        )
        self.btn_reset_zoom.pack(side="right", padx=6, pady=4)

        self.btn_zoom_out = ctk.CTkButton(
            self.map_toolbar,
            text="➖ Zoom -",
            width=75,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#30363d",
            hover_color="#484f58",
            command=lambda: self.pan_zoom.zoom_by_factor(1.3)
        )
        self.btn_zoom_out.pack(side="right", padx=4, pady=4)

        self.btn_zoom_in = ctk.CTkButton(
            self.map_toolbar,
            text="➕ Zoom +",
            width=75,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="#30363d",
            hover_color="#484f58",
            command=lambda: self.pan_zoom.zoom_by_factor(0.75)
        )
        self.btn_zoom_in.pack(side="right", padx=4, pady=4)

        # Canvas de Matplotlib
        self.fig = Figure(figsize=(8, 4.8), dpi=100, facecolor="#14181d")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0d1117")
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        self.ax.set_title("Malla Vial de Bogotá - Red Distrital BibloRed", color="#f0f6fc", fontsize=11, pad=8)
        self.ax.set_xlabel("Longitud (°)", color="#8b949e", fontsize=9)
        self.ax.set_ylabel("Latitud (°)", color="#8b949e", fontsize=9)
        self.ax.grid(True, color="#1c2430", linestyle="--", linewidth=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

        # Conectar el controlador de movimiento y zoom
        self.pan_zoom = MapPanZoomController(self.ax, self.canvas, on_reset=self._reset_map_view)

        # --- Subpanel 2: Cuadro de Menor Distancia y Tabla Comparativa ---
        self.metrics_container = ctk.CTkFrame(self.main_frame, corner_radius=8)
        self.metrics_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 8))

        # Cuadro destacado: Ganador Menor Distancia
        self.winner_card = ctk.CTkFrame(
            self.metrics_container,
            fg_color="#0e2a22",
            border_width=int(1.5),
            border_color="#06d6a0",
            corner_radius=8
        )
        self.winner_card.pack(fill="x", padx=12, pady=(6, 4))

        self.lbl_winner_title = ctk.CTkLabel(
            self.winner_card,
            text="🏆 GANADOR: MENOR DISTANCIA RECORRIDA",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#06d6a0"
        )
        self.lbl_winner_title.pack(padx=14, pady=(5, 2), anchor="w")

        self.lbl_winner_desc = ctk.CTkLabel(
            self.winner_card,
            text="Escribe o selecciona una biblioteca de origen y destino y haz clic en 'Calcular Menor Ruta' para evaluar los 4 modelos de búsqueda.",
            font=ctk.CTkFont(size=11),
            text_color="#d0f5ea",
            justify="left"
        )
        self.lbl_winner_desc.pack(padx=14, pady=(0, 5), anchor="w")

        # Scrollable Frame para la tabla comparativa de resultados
        self.table_frame = ctk.CTkScrollableFrame(self.metrics_container, height=130)
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        self._render_table_headers()

    def _render_table_headers(self):
        """Dibuja los encabezados de columnas de la tabla comparativa."""
        headers = ["Algoritmo / Modelo", "Tipo de Evaluación", "Distancia Recorrida (km)", "Nodos Evaluados", "Ruta", "Arbol de Estados", "Resultado / Estado"]
        col_weights = [3, 3, 2, 2, 1, 2, 2]

        for col_idx, (header, weight) in enumerate(zip(headers, col_weights)):
            self.table_frame.grid_columnconfigure(col_idx, weight=weight)
            lbl = ctk.CTkLabel(
                self.table_frame,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#58a6ff"
            )
            lbl.grid(row=0, column=col_idx, padx=6, pady=3, sticky="w")

    def _async_load_map(self):
        """Carga el mapa vehicular de Bogotá en caché en un hilo para no bloquear la interfaz."""
        self.is_loading_graph = True
        self.btn_search.configure(state="disabled")
        self.progress_bar.pack(padx=20, pady=(6, 3))
        self.progress_bar.start()
        self.lbl_status.configure(text="Cargando mapa vial de Bogotá...", text_color="#ffd166")

        def task():
            try:
                road_map = load_bogota_graph()
                self.after(0, lambda rm=road_map: self._on_map_loaded(rm))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._on_map_load_error(msg))

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def _on_map_loaded(self, road_map: BogotaRoadMap):
        """Callback invocado cuando el grafo vehicular ha sido cargado con éxito."""
        self.road_map = road_map
        self.is_loading_graph = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_search.configure(state="normal")
        self.lbl_status.configure(text="✅ Mapa listo. Puedes buscar bibliotecas y arrastrar el mapa.", text_color="#06d6a0")
        self.lbl_graph_info.configure(text=f"Red Vial: {road_map.total_nodes:,} nodos | {road_map.total_edges:,} vías")

        self._plot_libraries_overview()

    def _on_map_load_error(self, error_msg: str):
        """Callback invocado en caso de error durante la carga del mapa."""
        self.is_loading_graph = False
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.lbl_status.configure(text="❌ Error al cargar el mapa.", text_color="#ef476f")
        messagebox.showerror("Error de Carga", f"No se pudo cargar la red vial de Bogotá:\n\n{error_msg}")

    def _draw_road_network_background(self):
        """Dibuja el entramado de calles de Bogotá con LineCollection de alto rendimiento."""
        if self.road_map and self.road_map.street_lines:
            lc = LineCollection(
                self.road_map.street_lines,
                colors="#212a36",
                linewidths=0.55,
                alpha=0.6,
                zorder=1
            )
            self.ax.add_collection(lc)

    def _set_map_bounds(self, orig_name: str, dest_name: str):
        """Ajusta dinámicamente el zoom para enfocar las bibliotecas seleccionadas."""
        o_data = BIBLIOTECAS.get(orig_name)
        d_data = BIBLIOTECAS.get(dest_name)

        if o_data and d_data:
            min_lat = min(o_data["lat"], d_data["lat"])
            max_lat = max(o_data["lat"], d_data["lat"])
            min_lon = min(o_data["lon"], d_data["lon"])
            max_lon = max(o_data["lon"], d_data["lon"])

            margin_lat = max((max_lat - min_lat) * 0.25, 0.035)
            margin_lon = max((max_lon - min_lon) * 0.25, 0.045)

            self.ax.set_ylim(min_lat - margin_lat, max_lat + margin_lat)
            self.ax.set_xlim(min_lon - margin_lon, max_lon + margin_lon)
        else:
            self.ax.set_xlim(-74.22, -74.00)
            self.ax.set_ylim(4.50, 4.79)

    def _reset_map_view(self):
        """Reencuadra la vista del mapa al origen y destino seleccionados o al casco urbano."""
        orig = self.combo_origen.get()
        dest = self.combo_destino.get()
        self._set_map_bounds(orig, dest)
        self.canvas.draw_idle()

    def _plot_libraries_overview(self):
        """Renderiza el mapa de calles y la distribución de las bibliotecas de BibloRed."""
        self.ax.clear()
        self.ax.set_facecolor("#0d1117")
        self.ax.grid(True, color="#1c2430", linestyle="--", linewidth=0.5)
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        self.ax.set_title("Mapa Interactivo de la Red Vial y Bibliotecas Públicas - BibloRed", color="#f0f6fc", fontsize=11, pad=8)
        self.ax.set_xlabel("Longitud (°)", color="#8b949e", fontsize=9)
        self.ax.set_ylabel("Latitud (°)", color="#8b949e", fontsize=9)

        # 1. Dibujar red vial
        self._draw_road_network_background()

        # 2. Dibujar puntos de bibliotecas
        lats = [data["lat"] for data in BIBLIOTECAS.values()]
        lons = [data["lon"] for data in BIBLIOTECAS.values()]
        self.ax.scatter(lons, lats, c="#06d6a0", s=45, edgecolors="#ffffff", linewidth=0.8, alpha=0.9, zorder=4, label="Bibliotecas BibloRed")

        orig = self.combo_origen.get()
        dest = self.combo_destino.get()

        if orig in BIBLIOTECAS:
            o_lat, o_lon = BIBLIOTECAS[orig]["lat"], BIBLIOTECAS[orig]["lon"]
            self.ax.scatter([o_lon], [o_lat], c="#118ab2", s=130, edgecolors="#ffffff", linewidth=1.8, zorder=6, label=f"Origen: {orig[:22]}...")

        if dest in BIBLIOTECAS and dest != orig:
            d_lat, d_lon = BIBLIOTECAS[dest]["lat"], BIBLIOTECAS[dest]["lon"]
            self.ax.scatter([d_lon], [d_lat], c="#ef476f", s=130, edgecolors="#ffffff", linewidth=1.8, zorder=6, label=f"Destino: {dest[:22]}...")

        # 3. Ajustar límites de zoom
        self._set_map_bounds(orig, dest)

        self.ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_selection_change(self, _choice=None):
        """Actualiza la vista del mapa cuando el usuario cambia origen o destino en los buscadores."""
        if self.road_map is not None:
            self._plot_libraries_overview()

    def _on_search_clicked(self):
        """Valida entradas y ejecuta los algoritmos de búsqueda seleccionados en segundo plano."""
        road_map = self.road_map
        if road_map is None:
            messagebox.showwarning("Mapa no disponible", "La red vial aún no ha terminado de cargar.")
            return

        origen_name = self.combo_origen.get()
        destino_name = self.combo_destino.get()

        if origen_name not in BIBLIOTECAS:
            messagebox.showwarning("Origen no válido", f"'{origen_name}' no es una biblioteca válida de BibloRed.")
            return

        if destino_name not in BIBLIOTECAS:
            messagebox.showwarning("Destino no válido", f"'{destino_name}' no es una biblioteca válida de BibloRed.")
            return

        if origen_name == destino_name:
            messagebox.showinfo("Mismo Punto", "La biblioteca de origen y destino son la misma. La distancia es 0 metros.")
            return

        self.btn_search.configure(state="disabled")
        self.progress_bar.pack(padx=20, pady=(6, 3))
        self.progress_bar.start()
        self.lbl_status.configure(text="Buscando ruta con menor distancia...", text_color="#ffd166")

        algo_choice = self.combo_algo.get()

        def search_thread_func():
            try:
                # Nodos precomputados directos (O(1))
                if origen_name in road_map.library_nodes:
                    orig_node = road_map.library_nodes[origen_name]
                else:
                    orig_coords = (BIBLIOTECAS[origen_name]["lat"], BIBLIOTECAS[origen_name]["lon"])
                    orig_node = get_nearest_node(road_map, orig_coords[0], orig_coords[1])

                if destino_name in road_map.library_nodes:
                    dest_node = road_map.library_nodes[destino_name]
                else:
                    dest_coords = (BIBLIOTECAS[destino_name]["lat"], BIBLIOTECAS[destino_name]["lon"])
                    dest_node = get_nearest_node(road_map, dest_coords[0], dest_coords[1])

                # Lista de algoritmos con descripción precisa de su método de cálculo
                all_algos = [
                    ("4. Búsqueda A* (A-Star)", "Distancia Real g(n) + Heurística h(n)", astar_search, "#ef476f", True),
                    ("2. Costo Uniforme (UCS)", "Distancia Real Acumulada g(n)", ucs_search, "#06d6a0", True),
                    ("3. Búsqueda Voraz (Greedy)", "Solo Estimación h(n) en Línea Recta (No Real)", greedy_search, "#118ab2", False),
                    ("1. Búsqueda en Anchura (BFS)", "Conteo de Aristas / Giros (No Real)", bfs_search, "#ffd166", False)
                ]

                # Filtrar según la selección
                if "A*" in algo_choice:
                    selected_algos = [all_algos[0]]
                elif "UCS" in algo_choice:
                    selected_algos = [all_algos[1]]
                elif "Greedy" in algo_choice:
                    selected_algos = [all_algos[2]]
                elif "BFS" in algo_choice:
                    selected_algos = [all_algos[3]]
                else:
                    selected_algos = all_algos

                results = []
                for name, eval_type, func, color, uses_real in selected_algos:
                    tree_trace: List[Dict[str, Any]] = []
                    path, cost, nodes_eval, elapsed = func(
                        road_map, orig_node, dest_node, tree_trace=tree_trace
                    )
                    results.append({
                        "name": name,
                        "eval_type": eval_type,
                        "path": path,
                        "cost": cost,
                        "nodes_eval": nodes_eval,
                        "time_s": elapsed,
                        "tree_trace": tree_trace,
                        "color": color,
                        "uses_real": uses_real
                    })

                self.after(0, lambda res=results, o=origen_name, d=destino_name: self._on_search_finished(res, o, d))

            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._on_search_error(msg))

        thread = threading.Thread(target=search_thread_func, daemon=True)
        thread.start()

    def _on_search_error(self, message: str):
        """Restablece los controles e informa de errores al ejecutar una búsqueda."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_search.configure(state="normal")
        self.lbl_status.configure(text="❌ Error durante la búsqueda.", text_color="#ef476f")
        messagebox.showerror("Error de búsqueda", message)

    def _show_route_window(self, result: Dict[str, Any]):
        """Muestra en una ventana nativa los nodos y costos acumulados de una ruta."""
        path = result.get("path", [])
        if not path:
            messagebox.showinfo("Ruta no disponible", "Este algoritmo no encontró una ruta para mostrar.")
            return

        window = ctk.CTkToplevel(self)
        window.title(f"Ruta de {result['name']}")
        window.transient(self)
        window.resizable(True, True)

        width, height = 620, 560
        window.update_idletasks()
        x = self.winfo_rootx() + max((self.winfo_width() - width) // 2, 0)
        y = self.winfo_rooty() + max((self.winfo_height() - height) // 2, 0)
        window.geometry(f"{width}x{height}+{x}+{y}")

        ctk.CTkLabel(
            window,
            text=f"Ruta completa · {len(path):,} nodos",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#06d6a0"
        ).pack(padx=20, pady=(18, 4), anchor="w")
        ctk.CTkLabel(
            window,
            text=result["name"],
            font=ctk.CTkFont(size=12),
            text_color="#aeb8c1"
        ).pack(padx=20, pady=(0, 12), anchor="w")

        route_list = ctk.CTkScrollableFrame(window, corner_radius=6)
        route_list.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        accumulated_cost = 0.0

        for step_index, node in enumerate(path):
            if step_index > 0 and self.road_map is not None:
                accumulated_cost += get_edge_cost(self.road_map, path[step_index - 1], node)

            if step_index == 0:
                step_label = "Paso 1 · Origen"
            elif step_index == len(path) - 1:
                step_label = f"Paso {step_index + 1} · Destino"
            else:
                step_label = f"Paso {step_index + 1}"

            row = ctk.CTkFrame(route_list, fg_color="#18232c", corner_radius=5)
            row.pack(fill="x", padx=4, pady=3)
            ctk.CTkLabel(
                row,
                text=step_label,
                width=150,
                anchor="w",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#58a6ff"
            ).pack(side="left", padx=(10, 6), pady=8)
            ctk.CTkLabel(
                row,
                text=f"Intersección {node}",
                anchor="w",
                font=ctk.CTkFont(size=11)
            ).pack(side="left", fill="x", expand=True, padx=6, pady=8)
            ctk.CTkLabel(
                row,
                text=f"Acumulado: {accumulated_cost:,.1f} m",
                anchor="e",
                font=ctk.CTkFont(size=11),
                text_color="#c0c8d0"
            ).pack(side="right", padx=10, pady=8)

        ctk.CTkButton(
            window,
            text="Cerrar",
            width=120,
            command=window.destroy
        ).pack(padx=16, pady=(0, 16), anchor="e")
        window.grab_set()

    def _show_search_tree_window(self, result: Dict[str, Any]):
        """Dibuja hasta 200 estados del árbol de búsqueda con Matplotlib."""
        path = result.get("path", [])
        trace = result.get("tree_trace", [])
        if not path:
            messagebox.showinfo("Árbol no disponible", "No hay una ruta encontrada para graficar.")
            return

        max_nodes = 200
        displayed_path = path if len(path) <= max_nodes else path[:max_nodes - 1] + path[-1:]
        tree_nodes: List[Dict[str, Any]] = []
        path_child_ids: Dict[Tuple[int, str], int] = {}
        path_accumulated = 0.0

        for index, node in enumerate(displayed_path):
            if index > 0 and self.road_map is not None:
                prev_index = index - 1
                if len(path) > max_nodes and index == max_nodes - 1:
                    for edge_index in range(prev_index, len(path) - 1):
                        path_accumulated += get_edge_cost(
                            self.road_map, path[edge_index], path[edge_index + 1]
                        )
                else:
                    path_accumulated += get_edge_cost(self.road_map, path[prev_index], node)
            state_id = len(tree_nodes)
            parent_id = state_id - 1 if state_id else None
            tree_nodes.append({
                "id": state_id,
                "parent": parent_id,
                "node": str(node),
                "g": path_accumulated,
                "solution": True,
            })
            if parent_id is not None:
                path_child_ids[(parent_id, str(node))] = state_id

        trace_to_display: Dict[int, int] = {}
        if trace:
            trace_to_display[trace[0]["id"]] = 0
            for state in trace[1:]:
                parent_display_id = trace_to_display.get(state["parent"])
                if parent_display_id is None:
                    continue
                route_child = path_child_ids.get((parent_display_id, state["node"]))
                if route_child is not None:
                    trace_to_display[state["id"]] = route_child
                    continue
                if len(tree_nodes) >= max_nodes:
                    continue
                displayed_id = len(tree_nodes)
                tree_nodes.append({
                    "id": displayed_id,
                    "parent": parent_display_id,
                    "node": state["node"],
                    "g": state["g"],
                    "solution": False,
                })
                trace_to_display[state["id"]] = displayed_id

        children: Dict[int, List[int]] = {state["id"]: [] for state in tree_nodes}
        for state in tree_nodes:
            if state["parent"] is not None:
                children[state["parent"]].append(state["id"])

        positions: Dict[int, Tuple[float, float]] = {}
        leaf_counter = [0]

        def position_subtree(state_id: int, depth: int) -> float:
            child_ids = children[state_id]
            if not child_ids:
                x_position = float(leaf_counter[0])
                leaf_counter[0] += 1
            else:
                child_positions = [position_subtree(child_id, depth + 1) for child_id in child_ids]
                x_position = sum(child_positions) / len(child_positions)
            positions[state_id] = (x_position, float(-depth))
            return x_position

        position_subtree(0, 0)

        window = ctk.CTkToplevel(self)
        window.title(f"Árbol de Estados · {result['name']}")
        window.geometry("1120x760")
        window.minsize(760, 520)
        window.transient(self)

        ctk.CTkLabel(
            window,
            text=f"Árbol de búsqueda · {len(tree_nodes)} de máximo {max_nodes} nodos",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#58a6ff"
        ).pack(fill="x", padx=12, pady=(10, 4))

        figure = Figure(figsize=(11, 7), dpi=90, facecolor="#10161c")
        axis = figure.add_subplot(111)
        axis.set_facecolor("#10161c")
        axis.axis("off")
        axis.set_title(result["name"], color="#f0f6fc", fontsize=10, pad=12)

        for state in tree_nodes:
            parent_id = state["parent"]
            if parent_id is None:
                continue
            parent_x, parent_y = positions[parent_id]
            child_x, child_y = positions[state["id"]]
            edge_color = "#24c78b" if state["solution"] else "#677786"
            axis.plot([parent_x, child_x], [parent_y, child_y], color=edge_color, linewidth=1.1, zorder=1)

        for state in tree_nodes:
            x_position, y_position = positions[state["id"]]
            if state["id"] == 0:
                node_color = "#f0c84b"
            elif state["id"] == len(displayed_path) - 1:
                node_color = "#ef5b64"
            elif state["solution"]:
                node_color = "#24c78b"
            else:
                node_color = "#9bb5c9"
            axis.scatter([x_position], [y_position], s=115, color=node_color, edgecolors="#e8edf2", linewidths=0.7, zorder=2)
            axis.annotate(
                state["node"],
                (x_position, y_position),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#f0f6fc",
                fontsize=6,
                zorder=3
            )

        axis.margins(x=0.06, y=0.12)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="top", fill="x", padx=8)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

        ctk.CTkLabel(
            window,
            text="Origen · Amarillo     Destino · Rojo     Ruta final · Verde     Alternativas · Azul grisáceo",
            font=ctk.CTkFont(size=11),
            text_color="#c0c8d0"
        ).pack(padx=10, pady=3)
        ctk.CTkButton(window, text="Cerrar", width=110, command=window.destroy).pack(pady=(3, 12))
        window.grab_set()

    def _on_search_finished(self, results: List[Dict[str, Any]], origen_name: str, destino_name: str):
        """Actualiza el cuadro de menor distancia, la tabla de resultados y dibuja las rutas."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_search.configure(state="normal")
        self.lbl_status.configure(text="✅ Búsqueda completada. Puedes mover y hacer zoom en el mapa.", text_color="#06d6a0")
        self.last_results = results

        # Limpiar filas anteriores de la tabla
        for widget in self.table_row_widgets:
            widget.destroy()
        self.table_row_widgets.clear()

        # Identificar algoritmo con MENOR DISTANCIA RECORRIDA
        valid_results = [r for r in results if r["path"]]

        if valid_results:
            best_route = min(valid_results, key=lambda x: x["cost"])
            best_algo_name = best_route["name"].split(".")[1].split("(")[0].strip()
            best_distance_km = best_route["cost"] / 1000.0

            # Comparativa directa si está Voraz
            greedy_res = next((r for r in valid_results if "Voraz" in r["name"]), None)
            greedy_diff_text = ""
            if greedy_res and greedy_res != best_route:
                diff_m = greedy_res["cost"] - best_route["cost"]
                if diff_m > 0:
                    greedy_diff_text = f" | ⚡ Ahorro de {diff_m/1000.0:.2f} km frente a Búsqueda Voraz ({greedy_res['cost']/1000.0:.2f} km)"

            # Actualizar el Cuadro Destacado de Menor Distancia
            self.lbl_winner_title.configure(
                text=f"🏆 MENOR DISTANCIA RECORRIDA: {best_algo_name.upper()} ➔ {best_distance_km:.2f} km ({best_route['cost']:,.0f} m)",
                text_color="#06d6a0"
            )
            self.lbl_winner_desc.configure(
                text=(
                    f"⭐ {best_algo_name} calculó la ruta más corta óptima evaluando la distancia real vehicular acumulada g(n).\n"
                    f"📌 Ten en cuenta que Búsqueda Voraz solo se guía por la línea recta estimada h(n) y no evalúa la distancia real.{greedy_diff_text}"
                ),
                text_color="#d0f5ea"
            )
        else:
            self.lbl_winner_title.configure(
                text="❌ NO SE ENCONTRÓ RUTA VEHICULAR",
                text_color="#ef476f"
            )
            self.lbl_winner_desc.configure(
                text="No existe conectividad vial directa en el sentido vehicular entre los puntos seleccionados.",
                text_color="#ffd166"
            )

        # Poblar filas de resultados en la tabla
        min_cost = min((r["cost"] for r in valid_results), default=float("inf"))

        for row_idx, res in enumerate(results, start=1):
            name = res["name"]
            eval_type = res["eval_type"]
            cost_km = f"{res['cost'] / 1000.0:.2f} km" if res["cost"] != float("inf") else "Sin Ruta"
            nodes_cnt = f"{res['nodes_eval']:,}"
            # Badge / Resultado
            if not res["path"]:
                badge_text = "❌ No hallado"
                badge_color = "#ef476f"
            elif res["cost"] == min_cost:
                badge_text = "🏆 Menor Distancia (Óptimo)"
                badge_color = "#06d6a0"
            elif not res["uses_real"]:
                badge_text = "⚠️ Subóptimo (Sin dist. real)"
                badge_color = "#ffd166"
            else:
                badge_text = "✅ Ruta válida"
                badge_color = "#58a6ff"

            values = [name, eval_type, cost_km, nodes_cnt]
            for col_idx, val in enumerate(values):
                color = badge_color if col_idx == 2 else ("#ffffff" if col_idx == 0 else "#c0c8d0")
                lbl = ctk.CTkLabel(
                    self.table_frame,
                    text=val,
                    font=ctk.CTkFont(size=11, weight="bold" if col_idx == 2 and res["cost"] == min_cost else "normal"),
                    text_color=color
                )
                lbl.grid(row=row_idx, column=col_idx, padx=6, pady=3, sticky="w")
                self.table_row_widgets.append(lbl)

            route_button = ctk.CTkButton(
                self.table_frame,
                text="Ver Ruta 📍",
                width=112,
                height=27,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#245b45",
                hover_color="#2d7659",
                state="normal" if res["path"] else "disabled",
                command=lambda route_result=res: self._show_route_window(route_result)
            )
            route_button.grid(row=row_idx, column=4, padx=6, pady=3, sticky="w")
            self.table_row_widgets.append(route_button)

            tree_button = ctk.CTkButton(
                self.table_frame,
                text="Ver Grafico",
                width=105,
                height=27,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#315b78",
                hover_color="#3e7194",
                state="normal" if res["path"] else "disabled",
                command=lambda tree_result=res: self._show_search_tree_window(tree_result)
            )
            tree_button.grid(row=row_idx, column=5, padx=6, pady=3, sticky="w")
            self.table_row_widgets.append(tree_button)

            result_label = ctk.CTkLabel(
                self.table_frame,
                text=badge_text,
                font=ctk.CTkFont(size=11),
                text_color=badge_color
            )
            result_label.grid(row=row_idx, column=6, padx=6, pady=3, sticky="w")
            self.table_row_widgets.append(result_label)

        # Dibujar trazado de rutas en Matplotlib
        self._plot_calculated_routes(results, origen_name, destino_name)

    def _plot_calculated_routes(self, results: List[Dict[str, Any]], origen_name: str, destino_name: str):
        """Grafica el camino vehicular encontrado sobre el mapa de calles de Matplotlib."""
        road_map = self.road_map
        if road_map is None:
            return

        self.ax.clear()
        self.ax.set_facecolor("#0d1117")
        self.ax.grid(True, color="#1c2430", linestyle="--", linewidth=0.5)
        self.ax.tick_params(colors="#8b949e", labelsize=8)
        self.ax.set_title(f"Ruta Vehicular: {origen_name} ➔ {destino_name}", color="#f0f6fc", fontsize=11, pad=8)
        self.ax.set_xlabel("Longitud (°)", color="#8b949e", fontsize=9)
        self.ax.set_ylabel("Latitud (°)", color="#8b949e", fontsize=9)

        # 1. Dibujar red vial de fondo
        self._draw_road_network_background()

        # 2. Dibujar todas las bibliotecas de fondo
        all_lats = [data["lat"] for data in BIBLIOTECAS.values()]
        all_lons = [data["lon"] for data in BIBLIOTECAS.values()]
        self.ax.scatter(all_lons, all_lats, c="#444d56", s=30, alpha=0.7, zorder=2)

        # 3. Trazar rutas calculadas (priorizando visualmente la de Menor Distancia)
        valid_res = [r for r in results if r["path"]]
        min_cost = min((r["cost"] for r in valid_res), default=float("inf"))

        for res in results:
            path = res["path"]
            if not path:
                continue

            path_lats = [float(road_map.coords[n][0]) for n in path]
            path_lons = [float(road_map.coords[n][1]) for n in path]

            is_shortest = (res["cost"] == min_cost)
            line_width = 3.8 if is_shortest else 2.0
            alpha = 0.95 if is_shortest else 0.65
            zorder = 4 if is_shortest else 3

            self.ax.plot(
                path_lons,
                path_lats,
                color=res["color"],
                linewidth=line_width,
                alpha=alpha,
                label=f"{res['name'].split('.')[1].split('(')[0].strip()} ({res['cost']/1000:.2f} km)",
                zorder=zorder
            )

        # 4. Resaltar origen y destino
        o_data = BIBLIOTECAS[origen_name]
        d_data = BIBLIOTECAS[destino_name]

        self.ax.scatter([o_data["lon"]], [o_data["lat"]], c="#118ab2", s=150, edgecolors="#ffffff", linewidth=1.8, zorder=6, label=f"Origen: {origen_name[:18]}...")
        self.ax.scatter([d_data["lon"]], [d_data["lat"]], c="#ef476f", s=150, edgecolors="#ffffff", linewidth=1.8, zorder=6, label=f"Destino: {destino_name[:18]}...")

        # 5. Ajustar zoom dinámico
        self._set_map_bounds(origen_name, destino_name)

        self.ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw()
