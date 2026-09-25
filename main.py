"""
Punto de Entrada Principal (Main Entry Point)
BibloRed Bogotá - Sistema de Búsqueda de Rutas y Comparativa de Algoritmos Clásicos

Ejecuta la aplicación gráfica interactiva con CustomTkinter.
"""
import sys
from gui.interface import BibloRedApp


def main():
    """Función principal de inicio de la aplicación."""
    try:
        app = BibloRedApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\nAplicación interrumpida por el usuario.")
        sys.exit(0)


if __name__ == "__main__":
    main()
