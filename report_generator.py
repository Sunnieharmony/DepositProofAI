from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


def _text(value) -> str:
    return escape(str(value))


def _assessment_modes_summary(findings) -> str:
    labels = {
        "llm": "LLM explanation with rules-engine classification",
        "fallback_after_llm_error": "rules-based fallback after LLM failure",
        "rules_fallback_not_llm": "rules-only (LLM not used)",
    }
    counts = {}
    for finding in findings:
        mode = finding.get("mode", "rules_fallback_not_llm")
        label = labels.get(mode, "rules-based assessment")
        counts[label] = counts.get(label, 0) + 1
    return "; ".join(f"{count} {label}" for label, count in counts.items()) or "No deduction findings."


def _append_repair_benchmark(story, benchmark, styles) -> None:
    if not benchmark:
        return
    story.append(Paragraph(
        "Repair-price benchmark (context only; does not determine classification)",
        styles["Heading3"]
    ))
    story.append(Paragraph(
        f"<b>Comparison:</b> {_text(benchmark.get('match_description', 'Sample-rate comparison'))}; "
        f"{_text(benchmark.get('city', ''))}"
        + (f", {_text(benchmark['locality'])}" if benchmark.get("locality") else "")
        + f"; category: {_text(benchmark.get('category', 'not recorded'))}"
        + f"; {_text(benchmark.get('record_count', 0))} sample record(s).",
        styles["SmallText"]
    ))
    unit_suffix = f" per {_text(benchmark['unit'])}" if benchmark.get("unit") else ""
    story.append(Paragraph(
        f"<b>Sample range:</b> INR {float(benchmark.get('lower') or 0):,.2f}–"
        f"INR {float(benchmark.get('upper') or 0):,.2f}{unit_suffix}. "
        f"<b>Claim position:</b> {_text(benchmark.get('position', 'not stated'))}; "
        f"<b>difference from nearest boundary:</b> INR {float(benchmark.get('difference') or 0):,.2f}"
        + (
            f" ({float(benchmark['difference_percent']):.1f}%)"
            if benchmark.get("difference_percent") is not None and benchmark.get("position") != "within"
            else ""
        )
        + ".",
        styles["SmallText"]
    ))
    benchmark_rows = [[
        Paragraph("<b>City / locality</b>", styles["SmallText"]),
        Paragraph("<b>Category / item</b>", styles["SmallText"]),
        Paragraph("<b>Listed sample range</b>", styles["SmallText"]),
    ]]
    for row in benchmark.get("matched_records", [])[:8]:
        category_or_item = (
            row.get("category")
            or benchmark.get("category")
            or row.get("item", "")
        )
        benchmark_rows.append([
            Paragraph(_text(f"{row.get('city', '')}, {row.get('locality', '')}")[:300], styles["SmallText"]),
            Paragraph(_text(category_or_item)[:300], styles["SmallText"]),
            Paragraph(
                f"INR {float(row.get('minimum') or 0):,.2f}–INR {float(row.get('maximum') or 0):,.2f} "
                + (f"per {_text(row['unit'])}" if row.get("unit") else ""),
                styles["SmallText"],
            ),
        ])
    benchmark_table = Table(
        benchmark_rows,
        colWidths=[2.0*inch, 1.7*inch, 2.5*inch],
        repeatRows=1,
    )
    benchmark_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(benchmark_table)
    story.append(Paragraph(
        _text(benchmark.get(
            "sample_data_warning",
            "Sample data only; not a verified quotation or determination of actual cost.",
        )),
        styles["SmallText"],
    ))
    if benchmark.get("llm_explanation"):
        story.append(Paragraph(
            "<b>Optional hosted-model explanation:</b> " + _text(benchmark["llm_explanation"])
            + " (This does not change the calculation or audit classification.)",
            styles["SmallText"],
        ))


def generate_audit_pdf(case, findings, totals, reports_dir: Path, photo_pairs=None) -> str:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"depositproof_audit_{case['id']}_{uuid.uuid4().hex[:6]}.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        rightMargin=0.6*inch, leftMargin=0.6*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17365D")))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10, leading=15, textColor=colors.HexColor("#263445"),
        wordWrap="CJK", spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="SmallText", parent=styles["ReportBody"], fontSize=9, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="DeductionTitle", parent=styles["Heading2"], fontSize=18,
        leading=22, textColor=colors.HexColor("#17365D"), wordWrap="CJK",
        spaceBefore=12, spaceAfter=4, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading3"], fontSize=12,
        leading=16, textColor=colors.HexColor("#17365D"), spaceBefore=9,
        spaceAfter=5, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="StatusText", parent=styles["ReportBody"], fontSize=11,
        leading=15, textColor=colors.HexColor("#713F12"),
        backColor=colors.HexColor("#FEF3C7"),
        borderColor=colors.HexColor("#F2C66D"), borderWidth=0.8,
        borderPadding=8, spaceBefore=4, spaceAfter=9,
    ))
    styles.add(ParagraphStyle(
        name="PassageText", parent=styles["SmallText"],
        backColor=colors.HexColor("#F8FAFC"),
        borderColor=colors.HexColor("#DCE4EE"), borderWidth=0.6,
        borderPadding=8, spaceBefore=4, spaceAfter=6,
    ))
    story = [
        Paragraph("DepositProof AI", styles["TitleCenter"]),
        Paragraph("Rental Security Deposit Evidence Audit", styles["Heading2"]),
        Paragraph(f"<b>Case:</b> {_text(case['name'])} &nbsp;&nbsp; <b>Case ID:</b> {_text(case['id'])}", styles["BodyText"]),
        Spacer(1, 10),
    ]
    summary = [
        ["Metric", "Amount"],
        ["Security deposit", f"INR {case['deposit']:,.2f}"],
        ["Landlord-claimed deductions", f"INR {case['claimed_deductions']:,.2f}"],
        ["Proposed refund", "Not provided" if case["proposed_refund"] is None else f"INR {case['proposed_refund']:,.2f}"],
        ["Supported-classification total", f"INR {totals.get('Supported', 0):,.2f}"],
        ["Inconsistent-classification total", f"INR {totals.get('Inconsistent', 0):,.2f}"],
        ["Insufficiently documented total", f"INR {totals.get('Insufficiently Documented', 0):,.2f}"],
    ]
    table = Table(summary, colWidths=[3.8*inch, 2.2*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story += [
        table,
        Spacer(1, 8),
        Paragraph(f"<b>Assessment modes by deduction:</b> {_text(_assessment_modes_summary(findings))}", styles["SmallText"]),
        Spacer(1, 14),
    ]
    for i, f in enumerate(findings, 1):
        classification = f.get("classification", "Insufficiently Documented")
        story.append(Paragraph(
            f"{i:02d} — {_text(f.get('description', 'Deduction'))}",
            styles["DeductionTitle"],
        ))
        story.append(Paragraph(
            f"<b>Claimed amount:</b> INR {float(f.get('amount') or 0):,.2f}",
            styles["ReportBody"],
        ))
        story.append(Paragraph("Final classification", styles["SectionHeading"]))
        story.append(Paragraph(
            f"<b>Final classification:</b> {_text(classification)}",
            styles["StatusText"],
        ))
        assessment_details = [[
            Paragraph(
                "<b>Assessment method</b><br/>" + _text(f.get(
                    "assessment_method",
                    "Rules-based assessment; hosted LLM use not recorded.",
                )),
                styles["SmallText"],
            ),
            Paragraph(
                "<b>Rules-engine classification</b><br/>" + _text(f.get(
                    "rules_classification", classification,
                )),
                styles["SmallText"],
            ),
        ]]
        assessment_table = Table(assessment_details, colWidths=[3.65*inch, 3.65*inch])
        assessment_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#DCE4EE")),
            ("INNERGRID", (0,0), (-1,-1), 0.35, colors.HexColor("#DCE4EE")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("PADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(Paragraph("Assessment details", styles["SectionHeading"]))
        story.append(assessment_table)
        story.append(Paragraph("Explanation", styles["SectionHeading"]))
        explanation = str(f.get("explanation") or "No explanation available.")
        for paragraph in re.split(r"\n\s*\n", explanation.strip()):
            if paragraph.strip():
                story.append(Paragraph(
                    _text(paragraph).replace("\n", "<br/>"),
                    styles["ReportBody"],
                ))
        clauses = f.get("clauses") or []
        if clauses:
            story.append(Paragraph("Relevant agreement passages", styles["SectionHeading"]))
            for c in clauses[:5]:
                verified = "Verified" if c.get("quote_verified") else "Not verified"
                quote = _text(c.get("quote", "")).replace("\n", "<br/>")
                story.append(Paragraph(
                    f"<b>Source document:</b> {_text(c.get('filename', 'Agreement'))}<br/>"
                    f"<b>Page number:</b> {_text(c.get('page', 'unknown'))}<br/>"
                    f"<b>Clause text:</b><br/>{quote}<br/>"
                    f"<b>Quotation verification:</b> {verified}",
                    styles["PassageText"],
                ))
        story.append(Paragraph("Limitations", styles["SectionHeading"]))
        story.append(Paragraph(
            _text(f.get("limitations") or "Not provided."),
            styles["ReportBody"],
        ))
        photo = f.get("photo_observation") or {}
        if photo:
            if photo.get("mode") != "llm_analyzed":
                story.append(Paragraph(
                    "A photo pair is linked, but no successful vision-model observation is stored.",
                    styles["SmallText"]
                ))
            else:
                if photo.get("observation"):
                    story.append(Paragraph(f"<b>Summary:</b> {_text(photo['observation'])[:1500]}", styles["SmallText"]))
                story.append(Paragraph(
                    f"<b>Comparability:</b> {_text(photo.get('comparable', 'not stated'))}; "
                    f"same area visible in both: {_text(photo.get('same_area_visible_in_both', 'not stated'))}.",
                    styles["SmallText"]
                ))
                for key, label in (("move_in_condition", "Move-in condition"), ("move_out_condition", "Move-out condition")):
                    if photo.get(key):
                        story.append(Paragraph(f"<b>{label}:</b> {_text(photo[key])[:1000]}", styles["SmallText"]))

                changes = photo.get("observed_changes") or []
                if changes:
                    data = [[
                        Paragraph("<b>Observed feature / location</b>", styles["SmallText"]),
                        Paragraph("<b>Move-in condition</b>", styles["SmallText"]),
                        Paragraph("<b>Move-out condition</b>", styles["SmallText"]),
                        Paragraph("<b>Interpretation</b>", styles["SmallText"]),
                    ]]
                    for change in changes[:8]:
                        feature = change.get("feature") or "Visible feature"
                        if change.get("location"):
                            feature += f" ({change['location']})"
                        data.append([
                            Paragraph(_text(feature)[:400], styles["SmallText"]),
                            Paragraph(_text(change.get("move_in_condition") or "Not described")[:400], styles["SmallText"]),
                            Paragraph(_text(change.get("move_out_condition") or "Not described")[:400], styles["SmallText"]),
                            Paragraph(_text(change.get("interpretation") or "No interpretation provided")[:500], styles["SmallText"]),
                        ])
                    photo_table = Table(data, colWidths=[1.25*inch, 1.45*inch, 1.45*inch, 2.95*inch], repeatRows=1)
                    photo_table.setStyle(TableStyle([
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E2E8F0")),
                        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0,0), (-1,-1), "TOP"),
                        ("PADDING", (0,0), (-1,-1), 4),
                    ]))
                    story.append(photo_table)
                elif photo.get("visible_differences"):
                    story.append(Paragraph(
                        "<b>Observed changes:</b> " + _text("; ".join(photo["visible_differences"]))[:1200],
                        styles["SmallText"]
                    ))
                    story.append(Paragraph(
                        "The model did not provide separate move-in/move-out descriptions or locations.",
                        styles["SmallText"]
                    ))
                else:
                    story.append(Paragraph("<b>Observed changes:</b> No specific visible differences were reported.", styles["SmallText"]))

                uncertainty = photo.get("model_uncertainty")
                story.append(Paragraph(
                    f"<b>Evidence strength / uncertainty:</b> Listed changes are model-reported visual comparisons; "
                    f"possible-damage items are uncertain observations. Qualitative model uncertainty: "
                    f"{_text(uncertainty or 'not provided')}; this is not a calibrated confidence score.",
                    styles["SmallText"]
                ))
            possible_damage = photo.get("possible_damage_or_deterioration") or []
            if possible_damage:
                story.append(Paragraph(
                    "<b>Possible damage (not a finding of responsibility):</b> " +
                    _text("; ".join(possible_damage))[:1200],
                    styles["SmallText"]
                ))
            if photo.get("not_reliably_comparable"):
                story.append(Paragraph(
                    "<b>Not reliably comparable:</b> " + _text("; ".join(photo["not_reliably_comparable"]))[:1000],
                    styles["SmallText"]
                ))
            if photo.get("limitations"):
                story.append(Paragraph(f"<b>Model-stated limitations:</b> {_text(photo['limitations'])[:1000]}", styles["SmallText"]))
            story.append(Paragraph(
                f"<b>Relevance to deduction:</b> This stored photo comparison concerns "
                f"{_text(f.get('photo_item_name') or 'the linked photo item')}; it was not a deduction assessment. "
                f"It may provide visual context for “{_text(f.get('description', 'the claim'))}”, but does not "
                "establish that the claimed repair was required or its cost.",
                styles["SmallText"]
            ))
            story.append(Paragraph(
                "Photographs can show visible differences, but cannot independently establish when a change "
                "occurred, who caused it, actual repair cost, or legal responsibility.",
                styles["SmallText"]
            ))
        else:
            story.append(Paragraph(
                "No linked usable photo observation is available for this deduction.",
                styles["SmallText"]
            ))
        _append_repair_benchmark(story, f.get("repair_price_benchmark"), styles)
        if f.get("ai_error"):
            story.append(Paragraph(
                f"<b>Hosted AI unavailable:</b> {_text(f['ai_error'])[:1000]} (rules-based fallback used.)",
                styles["SmallText"]
            ))
        story.append(Spacer(1, 10))
    if photo_pairs:
        story.append(Paragraph("Saved photo comparisons not linked to a deduction", styles["Heading2"]))
        story.append(Paragraph(
            "These stored photo comparisons are not associated with a specific deduction and should not "
            "be treated as evidence for a particular claim.",
            styles["SmallText"]
        ))
        for pair in photo_pairs:
            photo = json.loads(pair["observation_json"]) if pair["observation_json"] else {
                "mode": "not_analyzed",
                "limitations": "No photo observation is stored.",
            }
            story.append(Paragraph(
                f"{_text(pair['item_name'])} — status: {_text(pair['status'])}",
                styles["Heading3"]
            ))
            if photo.get("mode") != "llm_analyzed":
                story.append(Paragraph(
                    "No successful vision-model observation is stored. "
                    + _text(photo.get("error") or photo.get("limitations") or "No analysis details available."),
                    styles["SmallText"]
                ))
                continue
            if photo.get("observation"):
                story.append(Paragraph(f"<b>Summary:</b> {_text(photo['observation'])[:1500]}", styles["SmallText"]))
            for key, label in (("move_in_condition", "Move-in condition"), ("move_out_condition", "Move-out condition")):
                if photo.get(key):
                    story.append(Paragraph(f"<b>{label}:</b> {_text(photo[key])[:1000]}", styles["SmallText"]))
            changes = photo.get("observed_changes") or []
            if changes:
                data = [[
                    Paragraph("<b>Observed feature / location</b>", styles["SmallText"]),
                    Paragraph("<b>Move-in condition</b>", styles["SmallText"]),
                    Paragraph("<b>Move-out condition</b>", styles["SmallText"]),
                    Paragraph("<b>Interpretation</b>", styles["SmallText"]),
                ]]
                for change in changes[:8]:
                    feature = change.get("feature") or "Visible feature"
                    if change.get("location"):
                        feature += f" ({change['location']})"
                    data.append([
                        Paragraph(_text(feature)[:400], styles["SmallText"]),
                        Paragraph(_text(change.get("move_in_condition") or "Not described")[:400], styles["SmallText"]),
                        Paragraph(_text(change.get("move_out_condition") or "Not described")[:400], styles["SmallText"]),
                        Paragraph(_text(change.get("interpretation") or "No interpretation provided")[:500], styles["SmallText"]),
                    ])
                table = Table(data, colWidths=[1.25*inch, 1.45*inch, 1.45*inch, 2.95*inch], repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E2E8F0")),
                    ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("PADDING", (0,0), (-1,-1), 4),
                ]))
                story.append(table)
            elif photo.get("visible_differences"):
                story.append(Paragraph(
                    "<b>Observed changes:</b> " + _text("; ".join(photo["visible_differences"]))[:1200],
                    styles["SmallText"]
                ))
            story.append(Paragraph(
                f"<b>Comparability:</b> {_text(photo.get('comparable', 'not stated'))}; "
                f"<b>qualitative uncertainty:</b> {_text(photo.get('model_uncertainty', 'not provided'))} "
                "(not a calibrated score).",
                styles["SmallText"]
            ))
            if photo.get("limitations"):
                story.append(Paragraph(f"<b>Limitations:</b> {_text(photo['limitations'])[:1000]}", styles["SmallText"]))
            story.append(Paragraph(
                "Photographs show visible differences only; they cannot independently establish timing, "
                "cause, repair cost, or legal responsibility.",
                styles["SmallText"]
            ))
    story += [
        Spacer(1, 12),
        Paragraph("Disclaimer", styles["Heading3"]),
        Paragraph(
            "This report combines rules-based evidence classifications with hosted-model explanations "
            "or photo comparisons only where explicitly indicated. It is not legal advice, "
            "a legal judgment, or a guarantee of refund. Missing documentation does not by itself "
            "prove a claim is invalid. Review original documents and seek qualified advice where appropriate.",
            styles["SmallText"]
        )
    ]
    doc.build(story)
    return str(path)
