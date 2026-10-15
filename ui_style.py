"""Shared flat, dark appearance for the two maze windows."""
from tkinter import ttk

BACKGROUND = "#101827"
FOREGROUND = "#e4edf4"


def apply_style(root):
    root.configure(background=BACKGROUND)
    style = ttk.Style(root)
    # Native macOS buttons ignore some color settings; use a controllable theme.
    style.theme_use("clam")
    style.configure("TFrame", background=BACKGROUND)
    style.configure("TLabel", background=BACKGROUND, foreground=FOREGROUND)
    style.configure("TCheckbutton", background=BACKGROUND, foreground=FOREGROUND)
    style.map("TCheckbutton", background=[("active", BACKGROUND)],
              foreground=[("active", "#91d8cc")])
    # Text-only buttons, with foreground feedback rather than filled backplates.
    style.layout("TButton", [("Button.padding", {"sticky": "nswe", "children": [
        ("Button.label", {"sticky": "nswe"})]})])
    style.configure("TButton", background=BACKGROUND, foreground="#91d8cc",
                    padding=(12, 9), borderwidth=0, relief="flat",
                    font=("Helvetica", 12))
    style.map("TButton", background=[("pressed", BACKGROUND),
              ("active", BACKGROUND), ("disabled", BACKGROUND)],
              foreground=[("disabled", "#65758c"), ("pressed", "#ffffff"),
                          ("active", "#ffffff"), ("focus", "#f5bd4f")])
    for name, value in {
        "background": BACKGROUND, "foreground": FOREGROUND,
        "activeBackground": "#91d8cc", "troughColor": "#26344a",
        "highlightThickness": 0, "borderWidth": 0, "relief": "flat",
    }.items():
        root.option_add(f"*Scale.{name}", value)
