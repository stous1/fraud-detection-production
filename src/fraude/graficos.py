"""Gráficos con Altair, guardados en HTML y PNG (vl-convert).

Paleta del portafolio (la misma del proyecto 6), validada con el validador de paletas: pasa,
pero rojo y verde quedan cerca para deuteranopía y el naranja tiene poco contraste, así que
nunca van juntos sin etiqueta directa. Todas las marcas llevan tooltip en el HTML.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert as vlc

AZUL = "#2f6db5"
NARANJA = "#d9822b"
GRIS = "#8a8f98"
VERDE = "#2e9e6b"
ROJO = "#c8453b"
MORADO = "#7b5ea7"
PALETA = [AZUL, NARANJA, MORADO, VERDE, GRIS]

# Números en español: coma decimal y punto de miles.
LOCALE = {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["", " US$"]}

alt.data_transformers.disable_max_rows()


@alt.theme.register("fraude", enable=True)
def _tema() -> alt.theme.ThemeConfig:
    return {
        "config": {
            "background": "#fcfcfb",
            "font": "Segoe UI, Helvetica, Arial, sans-serif",
            "view": {"stroke": None},
            "axis": {
                "gridColor": "#e6e6e3",
                "domainColor": "#b5b5b0",
                "tickColor": "#b5b5b0",
                "labelColor": "#4a4a46",
                "titleColor": "#4a4a46",
                "titleFontWeight": "normal",
            },
            "title": {"anchor": "start", "fontSize": 14, "color": "#1f1f1c", "offset": 10},
            "legend": {"labelColor": "#4a4a46", "titleColor": "#4a4a46", "orient": "top"},
        }
    }


def guardar(grafico: alt.TopLevelMixin, ruta_sin_ext: Path) -> None:
    """Escribe ``.html`` y ``.png`` (escala 2) del gráfico."""
    ruta_sin_ext.parent.mkdir(parents=True, exist_ok=True)
    grafico.save(str(ruta_sin_ext.with_suffix(".html")), embed_options={"formatLocale": LOCALE})
    png = vlc.vegalite_to_png(grafico.to_dict(), scale=2, format_locale=LOCALE)
    ruta_sin_ext.with_suffix(".png").write_bytes(png)


def _eje(formato: str) -> alt.Axis:
    return alt.Axis(format=formato, tickCount=6) if formato else alt.Axis(tickCount=6)


def barras_h(
    df: pd.DataFrame,
    x: str,
    y: str,
    titulo: str = "",
    eje_x: str = "",
    formato_x: str = "",
    color: str | None = None,
    orden: list[str] | None = None,
    ancho: int = 460,
    tooltip: list[str] | None = None,
) -> alt.LayerChart:
    """Barras horizontales con el valor escrito al final de cada barra."""
    enc_y = alt.Y(f"{y}:N", sort=orden if orden is not None else "-x", title=None)
    col = (
        alt.Color(f"{color}:N", title=None, scale=alt.Scale(range=PALETA))
        if color
        else alt.value(AZUL)
    )
    base = alt.Chart(df).encode(
        y=enc_y,
        x=alt.X(f"{x}:Q", title=eje_x, axis=_eje(formato_x)),
        tooltip=tooltip or [y, alt.Tooltip(f"{x}:Q", format=formato_x or ",.3f")],
    )
    barras = base.mark_bar(cornerRadiusEnd=4, height=14).encode(color=col)
    texto = base.mark_text(align="left", dx=4, color="#4a4a46").encode(
        text=alt.Text(f"{x}:Q", format=formato_x or ",.3f")
    )
    alto = max(100, 24 * df[y].nunique())
    return (barras + texto).properties(title=titulo, width=ancho, height=alto)


def barras_agrupadas(
    df: pd.DataFrame,
    x: str,
    y: str,
    grupo: str,
    titulo: str = "",
    eje_x: str = "",
    formato_x: str = "",
    orden_y: list[str] | None = None,
    orden_grupo: list[str] | None = None,
    ancho: int = 420,
) -> alt.FacetChart:
    """Un panel de barras horizontales por valor de ``grupo`` (en lugar de colores)."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort=orden_y, title=None, axis=alt.Axis(labelLimit=360)),
        x=alt.X(f"{x}:Q", title=eje_x, axis=_eje(formato_x)),
        tooltip=[grupo, y, alt.Tooltip(f"{x}:Q", format=formato_x or ",.3f")],
    )
    capas = base.mark_bar(cornerRadiusEnd=4, height=12, color=AZUL) + base.mark_text(
        align="left", dx=4, color="#4a4a46"
    ).encode(text=alt.Text(f"{x}:Q", format=formato_x or ",.3f"))
    alto = max(80, 22 * df[y].nunique())
    return (
        capas.properties(width=ancho, height=alto)
        .facet(row=alt.Row(f"{grupo}:N", sort=orden_grupo, title=None))
        .properties(title=titulo)
    )


def lineas(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    titulo: str = "",
    eje_x: str = "",
    eje_y: str = "",
    formato_y: str = "",
    ancho: int = 560,
    alto: int = 240,
    sombra: tuple[float, float] | None = None,
    etiqueta_sombra: str = "",
    cero: bool = True,
) -> alt.LayerChart:
    """Líneas de 2 px con puntos; ``sombra`` marca un intervalo del eje x."""
    enc = {
        "x": alt.X(f"{x}:Q", title=eje_x, scale=alt.Scale(nice=False, zero=False)),
        "y": alt.Y(f"{y}:Q", title=eje_y, axis=_eje(formato_y), scale=alt.Scale(zero=cero)),
        "tooltip": [*([color] if color else []), x, alt.Tooltip(f"{y}:Q", format=formato_y)],
    }
    if color:
        enc["color"] = alt.Color(f"{color}:N", title=None, scale=alt.Scale(range=PALETA))
    base = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36))
    capas: list[alt.Chart] = [
        base.encode(**enc) if color else base.encode(**enc, color=alt.value(AZUL))
    ]
    if sombra is not None:
        caja = pd.DataFrame({"a": [sombra[0]], "b": [sombra[1]], "t": [etiqueta_sombra]})
        capas.insert(
            0,
            alt.Chart(caja).mark_rect(opacity=0.10, color=GRIS).encode(x="a:Q", x2="b:Q"),
        )
        if etiqueta_sombra:
            capas.append(
                alt.Chart(caja)
                .mark_text(align="left", baseline="top", dx=4, dy=4, color="#6a6a66")
                .encode(x="a:Q", y=alt.value(0), text="t:N")
            )
    return alt.layer(*capas).properties(title=titulo, width=ancho, height=alto)


def barras_v(
    df: pd.DataFrame,
    x: str,
    y: str,
    titulo: str = "",
    eje_x: str = "",
    eje_y: str = "",
    formato_y: str = "",
    color: str | None = None,
    orden_color: list[str] | None = None,
    ancho: int = 560,
    alto: int = 220,
) -> alt.Chart:
    """Barras verticales (por ejemplo, volumen por semana coloreado por periodo)."""
    enc = {
        "x": alt.X(f"{x}:O", title=eje_x, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        "y": alt.Y(f"{y}:Q", title=eje_y, axis=_eje(formato_y)),
        "tooltip": [*([color] if color else []), x, alt.Tooltip(f"{y}:Q", format=formato_y)],
    }
    if color:
        enc["color"] = alt.Color(
            f"{color}:N", title=None, sort=orden_color, scale=alt.Scale(range=PALETA)
        )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(**enc, **({} if color else {"color": alt.value(AZUL)}))
        .properties(title=titulo, width=ancho, height=alto)
    )
