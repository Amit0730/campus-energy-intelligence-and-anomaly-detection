"""Build the downloadable assessment evidence from the active dashboard run."""

import json
import pandas as pd
from .day_ahead import OUTPUT, TIME
from .incidents import group_incidents, investigation


def sections():
    exp = json.loads((OUTPUT / "experiment.json").read_text())
    real = json.loads((OUTPUT / "real_meter.json").read_text())
    data = pd.read_csv(
        OUTPUT / "evaluation.csv.gz", parse_dates=[TIME], float_precision="round_trip"
    )
    incidents = group_incidents(data, exp)
    example_hour = pd.Timestamp("2025-09-25 02:00")
    example = incidents[
        (incidents.building == "Academic_Block_1")
        & (incidents.start <= example_hour)
        & (incidents.end >= example_hour)
    ].iloc[0]
    selected = exp["winner"]
    benchmark = exp["test"][selected]
    return [
        (
            "Project scope",
            "Campus energy intelligence",
            "A facilities reviewer selects a building, checks its expected electricity use, investigates an alert and records the inspection.",
            ["Evidence", "Scope"],
            [
                [
                    "Campus data",
                    "Five simulated facilities with hourly readings in 2024 and 2025.",
                ],
                [
                    "Forecast task",
                    "Predict 24 hourly kWh values at midnight for each facility.",
                ],
                [
                    "Independent check",
                    "A separately trained forecasting model uses measured BDG2 electricity data.",
                ],
                [
                    "Workflow",
                    "Campus overview, Building explorer, Forecast lab and Incident desk.",
                ],
            ],
        ),
        (
            "Experiment design",
            "Forecasts use information available at midnight",
            "Features combine calendar, floor area, yesterday’s temperature, last week’s occupancy, previous daily profiles and prior-day mean energy. Missing lag history is excluded.",
            ["Chronological period", "Dates", "Building-hours"],
            [
                [k, " to ".join(v), f"{exp['counts'][k]:,}"]
                for k, v in exp["dates"].items()
            ],
        ),
        (
            "Forecasting results",
            "How much does the forecast improve?",
            f"{selected} had the lowest validation RMSE. Its held-out hourly MAE was {benchmark['MAE']:.2f} kWh. Model selection and interval calibration precede the test period.",
            ["Model", "Validation RMSE", "Test RMSE", "Test MAE"],
            [
                [
                    k,
                    f"{exp['validation'][k]['RMSE']:.3f}",
                    f"{v['RMSE']:.3f}",
                    f"{v['MAE']:.3f}",
                ]
                for k, v in exp["test"].items()
            ],
        ),
        (
            "Features and uncertainty",
            "Which inputs change the error?",
            f"Removing one group at a time tests its incremental value. The nominal 80% range covered {exp['coverage']:.1%} of test readings. Coverage varies with building and season.",
            ["XGBoost features", "Test RMSE (kWh)", "Test MAE (kWh)"],
            [
                [k, f"{v['RMSE']:.3f}", f"{v['MAE']:.3f}"]
                for k, v in exp["ablation"].items()
            ],
        ),
        (
            "Anomaly detection",
            "How many labelled faults were missed?",
            "Thresholds use earlier calibration data per building. The combined method averages the ranked forecast deviation and Isolation Forest score. Each count represents a building-hour.",
            ["Method", "Precision", "Recall", "False alarms", "Missed"],
            [
                [
                    k,
                    f"{v['Precision']:.1%}",
                    f"{v['Recall']:.1%}",
                    str(v["False alarms"]),
                    str(v["Missed"]),
                ]
                for k, v in exp["detectors"].items()
            ],
        ),
        (
            "Case investigation",
            f"An overnight alert in {example.building.replace('_', ' ')}",
            investigation(example)
            + " The signed forecast difference estimates the size of the departure. Confirmation requires an inspection.",
            ["Evidence", "Observation"],
            [
                ["First flagged reading", str(example.start)],
                ["Duration", f"{example.hours} hour(s)"],
                [
                    "Actual / expected",
                    f"{example.actual:.2f} / {example.expected:.2f} kWh",
                ],
                ["Difference", f"{example.difference:+.2f} kWh"],
                ["Mean occupancy", f"{example.occupancy:.1%}"],
            ],
        ),
        (
            "External benchmark",
            "Testing one measured education-building meter",
            f"BDG2 meter {real['building']}. Training uses {real['training_hours']:,} valid hours in 2016. Testing uses {real['test_hours']:,} in 2017. Calendar and prior meter readings are available at the forecast origin. This meter has no verified equipment fault labels.",
            ["Model", "Test RMSE (kWh)", "Test MAE (kWh)"],
            [
                [k, f"{v['RMSE']:.3f}", f"{v['MAE']:.3f}"]
                for k, v in real["scores"].items()
            ],
        ),
    ]


def export():
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Table,
        TableStyle,
        Spacer,
        PageBreak,
    )
    from xml.sax.saxutils import escape

    dest = OUTPUT / "assessment"
    dest.mkdir(exist_ok=True)
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.33), Inches(7.5)
    ink = RGBColor.from_string("163830")
    green = RGBColor.from_string("006E5C")

    def textbox(slide, x, y, w, h, content, size, color, bold=False):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        box.text_frame.word_wrap = True
        p = box.text_frame.paragraphs[0]
        p.text = content
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "CampusTitle", fontSize=28, leading=34, textColor="#163830", spaceAfter=12
    )
    paragraph = ParagraphStyle(
        "CampusBody", fontSize=13, leading=19, textColor="#283e37", spaceAfter=17
    )
    flow = []
    for index, (label, heading, body, headers, rows) in enumerate(sections(), 1):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        textbox(slide, 0.62, 0.35, 12, 0.4, label.upper(), 14, green, True)
        textbox(slide, 0.62, 0.88, 12.05, 0.96, heading, 32, ink, True)
        textbox(slide, 0.65, 1.88, 11.9, 1.12, body, 17, ink)
        widths = [0.31] + [0.69 / (len(headers) - 1)] * (len(headers) - 1)
        table = slide.shapes.add_table(
            len(rows) + 1,
            len(headers),
            Inches(0.7),
            Inches(3.12),
            Inches(11.85),
            Inches(0.5 * (len(rows) + 1)),
        ).table
        for j, pct in enumerate(widths):
            table.columns[j].width = Inches(11.85 * pct)
        for i, row in enumerate([headers] + rows):
            for j, val in enumerate(row):
                cell = table.cell(i, j)
                cell.fill.solid()
                cell.fill.fore_color.rgb = (
                    green
                    if i == 0
                    else RGBColor.from_string("EDF5F1" if i % 2 else "F7FAF9")
                )
                cell.text = str(val)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                p = cell.text_frame.paragraphs[0]
                p.font.name = "Aptos"
                p.font.size = Pt(15)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor(255, 255, 255) if i == 0 else ink
        foot = (
            "Miller et al. (2020) · doi.org/10.1038/s41597-020-00712-x"
            if label == "External benchmark"
            else "Campus Energy Consumption Intelligence & Anomaly Detection · INT395"
        )
        textbox(slide, 0.65, 6.95, 11, 0.3, foot, 11, ink)
        textbox(slide, 12.1, 6.95, 0.5, 0.3, str(index), 11, ink)
        if flow:
            flow.append(PageBreak())
        flow.extend(
            [
                Paragraph(escape(label.upper()), styles["Heading3"]),
                Paragraph(escape(heading), title),
                Paragraph(escape(body), paragraph),
            ]
        )
        cellstyle = ParagraphStyle("cell", fontSize=11, leading=15, textColor="#163830")
        headstyle = ParagraphStyle(
            "head", parent=cellstyle, textColor=colors.white, fontName="Helvetica-Bold"
        )
        tbl = Table(
            [
                [
                    Paragraph(escape(str(v)), headstyle if n == 0 else cellstyle)
                    for v in row
                ]
                for n, row in enumerate([headers] + rows)
            ],
            colWidths=[740 * pct for pct in widths],
            hAlign="LEFT",
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#006E5C")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#EDF5F1"), colors.white],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            )
        )
        flow.extend([tbl, Spacer(1, 22), Paragraph(escape(foot), styles["Normal"])])
    deck.save(dest / "Campus_Energy_Viva.pptx")
    pdf = SimpleDocTemplate(
        str(dest / "Campus_Energy_Report.pdf"),
        pagesize=landscape(A4),
        leftMargin=46,
        rightMargin=46,
        topMargin=45,
        bottomMargin=34,
    )
    pdf.build(flow)


if __name__ == "__main__":
    export()
