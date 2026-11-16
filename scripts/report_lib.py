"""Helper reportlab untuk laporan PDF (angka selalu dibaca dari artifacts/*.json)."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FONT, FONT_B, FONT_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
_DJ = "/usr/share/fonts/truetype/dejavu/"
if os.path.exists(_DJ + "DejaVuSans.ttf"):
    pdfmetrics.registerFont(TTFont("DejaVu", _DJ + "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", _DJ + "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", _DJ + "DejaVuSans-Oblique.ttf"))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                       boldItalic="DejaVu-Bold")
    FONT, FONT_B, FONT_I = "DejaVu", "DejaVu-Bold", "DejaVu-Oblique"

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1b1b1a")
INK2 = colors.HexColor("#52514e")
LIGHT = colors.HexColor("#eef4fc")
GRIDC = colors.HexColor("#d4d3cf")

MEMBERS = [("04231011", "Arief Rahmansyah Firdaus"), ("04231018", "Davin Oentario Adytiawarman"),
           ("04231065", "Musta'in Yusuf"), ("04231063", "Muhammad Rozul Faza")]

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", fontName=FONT_B, fontSize=17, leading=21, textColor=INK, alignment=TA_CENTER,
                            spaceAfter=4),
    "sub": ParagraphStyle("s", fontName=FONT, fontSize=10.5, leading=14, textColor=INK2, alignment=TA_CENTER,
                          spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName=FONT_B, fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10,
                         spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName=FONT_B, fontSize=10.5, leading=14, textColor=INK, spaceBefore=7,
                         spaceAfter=3),
    "p": ParagraphStyle("p", fontName=FONT, fontSize=9.3, leading=13.2, textColor=INK, alignment=TA_JUSTIFY,
                        spaceAfter=4),
    "small": ParagraphStyle("sm", fontName=FONT, fontSize=7.8, leading=10, textColor=INK2),
    "cap": ParagraphStyle("cap", fontName=FONT_I, fontSize=8, leading=10, textColor=INK2, alignment=TA_CENTER,
                          spaceAfter=7),
    "cell": ParagraphStyle("c", fontName=FONT, fontSize=7.8, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName=FONT_B, fontSize=7.8, leading=9.6, textColor=colors.white),
    "formula": ParagraphStyle("f", fontName=FONT, fontSize=9.3, leading=13, textColor=INK, alignment=TA_CENTER,
                              backColor=colors.HexColor("#f5f5f2"), borderPadding=5, spaceBefore=3,
                              spaceAfter=7),
}


def f4(x):
    return f"{x:.4f}"


def pm(d, k="mean", s="std"):
    return f"{d[k]:.4f} ± {d[s]:.4f}"


class Report:
    def __init__(self, path: Path, title: str, subtitle: str, week: str):
        self.path, self.title, self.week = Path(path), title, week
        self.story = []
        self.fig_n = 0
        self.tab_n = 0
        self.story += [Paragraph(title, S["title"]), Paragraph(subtitle, S["sub"])]
        info = [["Mata kuliah", "Kecerdasan Buatan (TE2514031) · Gasal 2026/2027"],
                ["Pengampu", "Ir. Muhammad Ridho Dewanto, S.T., M.T."],
                ["Kelompok", "Kelas B · Kelompok 15 · Smart Room Occupancy Estimation"],
                ["Anggota", "; ".join(f"{n} ({i})" for i, n in MEMBERS)],
                ["Dataset", "UCI Room Occupancy Estimation (ID 864) · RNN · Fuzzy HVAC"],
                ["Tanggal", date.today().strftime("%d %B %Y")]]
        t = Table([[Paragraph(f"<b>{a}</b>", S["cell"]), Paragraph(b, S["cell"])] for a, b in info],
                  colWidths=[3.0 * cm, 14.2 * cm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT), ("GRID", (0, 0), (-1, -1), 0.4, GRIDC),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        self.story += [t, Spacer(1, 6)]

    def h1(self, t):
        self.story.append(Paragraph(t, S["h1"]))

    def h2(self, t):
        self.story.append(Paragraph(t, S["h2"]))

    def p(self, t):
        self.story.append(Paragraph(t, S["p"]))

    def formula(self, t):
        self.story.append(Paragraph(t, S["formula"]))

    def bullets(self, items):
        self.story.append(ListFlowable([ListItem(Paragraph(i, S["p"]), leftIndent=10) for i in items],
                                       bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8))

    def table(self, header, rows, widths=None, caption=None, zebra=True, font_size=None):
        st = S["cell"] if not font_size else ParagraphStyle("cx", parent=S["cell"], fontSize=font_size,
                                                             leading=font_size * 1.25)
        sth = S["cellb"] if not font_size else ParagraphStyle("cbx", parent=S["cellb"], fontSize=font_size,
                                                               leading=font_size * 1.25)
        data = [[Paragraph(str(h), sth) for h in header]]
        data += [[Paragraph(str(c), st) for c in r] for r in rows]
        if widths:
            widths = [w * cm for w in widths]
        t = Table(data, colWidths=widths, repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("GRID", (0, 0), (-1, -1), 0.4, GRIDC),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if zebra:
            for i in range(1, len(data)):
                if i % 2 == 0:
                    style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fb")))
        t.setStyle(TableStyle(style))
        items = []
        if caption:
            self.tab_n += 1
            items.append(Paragraph(f"Tabel {self.tab_n}. {caption}", S["cap"]))
        items.append(t)
        items.append(Spacer(1, 7))
        self.story.append(KeepTogether(items))

    def img(self, path, width_cm=16.5, caption=None):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
        width = width_cm * cm
        items = [Image(str(path), width=width, height=width * h / w)]
        if caption:
            self.fig_n += 1
            items.append(Paragraph(f"Gambar {self.fig_n}. {caption}", S["cap"]))
        self.story.append(KeepTogether(items))

    def pagebreak(self):
        self.story.append(PageBreak())

    def space(self, h=6):
        self.story.append(Spacer(1, h))

    def build(self):
        week = self.week

        def deco(canvas, doc):
            canvas.saveState()
            canvas.setFont(FONT, 7.5)
            canvas.setFillColor(INK2)
            canvas.drawString(2 * cm, A4[1] - 1.2 * cm, f"TE2514031 Kecerdasan Buatan · Tugas Besar · Kelas B Kelompok 15 · {week}")
            canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Halaman {doc.page}")
            canvas.setStrokeColor(GRIDC)
            canvas.line(2 * cm, A4[1] - 1.35 * cm, A4[0] - 2 * cm, A4[1] - 1.35 * cm)
            canvas.restoreState()

        doc = SimpleDocTemplate(str(self.path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                                topMargin=1.8 * cm, bottomMargin=1.7 * cm, title=self.title,
                                author="Kelas B Kelompok 15")
        doc.build(self.story, onFirstPage=deco, onLaterPages=deco)
        return self.path
