"""utils/pdf_generator.py – Generate PDF patient report using ReportLab."""

import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_patient_pdf(prediction_data: dict, username: str) -> bytes:
    """
    Generate a styled PDF report for a single prediction record.
    Returns raw PDF bytes.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=50, bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ──────────────────────────────────────────
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#1a73e8'),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#5f6368'),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1a73e8'),
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=16,
        textColor=colors.HexColor('#202124'),
    )
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Normal'],
        fontSize=16,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    # ── Result colours ─────────────────────────────────────────
    is_diabetic   = prediction_data.get('result', 0) == 1
    result_label  = "DIABETIC" if is_diabetic else "NOT DIABETIC"
    result_color  = colors.HexColor('#ea4335') if is_diabetic else colors.HexColor('#34a853')
    risk_pct      = prediction_data.get('risk_percentage', 0)

    # ── Build flowables ────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("🩺 Diabetes Disease Prediction System", title_style))
    story.append(Paragraph("Patient Diagnostic Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a73e8')))
    story.append(Spacer(1, 12))

    # Report meta
    import datetime as dt
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    now = dt.datetime.now(IST).strftime("%B %d, %Y  %I:%M %p")
    meta_data = [
        ["Report Generated:", now],
        ["Patient / User:", username],
        ["Report ID:", f"RPT-{prediction_data.get('id', 'N/A'):05d}"],
        ["Prediction Date:", prediction_data.get('timestamp', 'N/A')],
    ]
    meta_table = Table(meta_data, colWidths=[160, 340])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',  (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',  (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',  (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#5f6368')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Result box
    result_style_colored = ParagraphStyle(
        'ResultColored',
        parent=result_style,
        textColor=result_color,
    )
    story.append(Paragraph(f"Prediction Result: {result_label}", result_style_colored))
    story.append(Paragraph(
        f"<b>Risk Probability: {risk_pct}%</b>",
        ParagraphStyle('RiskPct', parent=body_style, fontSize=12, alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dadce0')))

    # Clinical Input Values
    story.append(Paragraph("Clinical Input Parameters", section_style))
    params = [
        ["Parameter", "Value", "Reference Range"],
        ["Pregnancies",              prediction_data.get('pregnancies', '-'),     "0 – 17"],
        ["Glucose (mg/dL)",          prediction_data.get('glucose', '-'),          "70 – 140"],
        ["Blood Pressure (mm Hg)",   prediction_data.get('blood_pressure', '-'),   "60 – 90"],
        ["Skin Thickness (mm)",      prediction_data.get('skin_thickness', '-'),   "10 – 50"],
        ["Insulin (µU/mL)",          prediction_data.get('insulin', '-'),          "16 – 166"],
        ["BMI (kg/m²)",              prediction_data.get('bmi', '-'),              "18.5 – 24.9"],
        ["Diabetes Pedigree Func.",  prediction_data.get('diabetes_pedigree', '-'), "0.08 – 2.42"],
        ["Age (years)",              prediction_data.get('age', '-'),              "21 – 81"],
    ]
    param_table = Table(params, colWidths=[200, 120, 180])
    param_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0),  colors.HexColor('#1a73e8')),
        ('TEXTCOLOR',    (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',     (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9fa'), colors.white]),
        ('ALIGN',        (1, 0), (-1, -1), 'CENTER'),
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#dadce0')),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 6),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 10))

    # Interpretation
    story.append(Paragraph("Result Interpretation", section_style))
    interp = prediction_data.get('interpretation',
             "Please consult your healthcare provider for a full interpretation.")
    story.append(Paragraph(interp, body_style))
    story.append(Spacer(1, 10))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dadce0')))
    story.append(Spacer(1, 6))
    disclaimer = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#9aa0a6'), alignment=TA_CENTER
    )
    story.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI-powered system and is for informational "
        "purposes only. It does NOT constitute medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional.", disclaimer
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
