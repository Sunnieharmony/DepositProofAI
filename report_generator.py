from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)


_BORDER = colors.HexColor("#E2E8F0")
_MUTED_ROW = colors.HexColor("#F8FAFC")
_STATUS_COLORS = {
    "Supported": (colors.HexColor("#DCFCE7"), colors.HexColor("#166534")),
    "Insufficiently Documented": (
        colors.HexColor("#FEF3C7"),
        colors.HexColor("#92400E"),
    ),
    "Inconsistent": (colors.HexColor("#FEE2E2"), colors.HexColor("#991B1B")),
}


def _text(value) -> str:
    return escape(str(value))


def _money(value) -> str:
    return f"INR {float(value or 0):,.2f}"


def _title_case(value) -> str:
    return " ".join(str(value or "Deduction").split()).title()


def _report_limitations(value) -> str:
    limitations = str(value or "Not provided.")
    if re.search(r"\brules[- ]engine\b|\bclassification rules\b", limitations, re.IGNORECASE):
        return (
            "This assessment is limited to the evidence provided and does not determine "
            "responsibility or legal liability."
        )
    return limitations


def _is_yes(value) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"true", "yes"}
    )


def _photo_match_text(photo) -> str:
    if photo.get("not_reliably_comparable"):
        return "Not verified (the images were reported as not reliably comparable)."

    comparable = photo.get("comparable")
    same_area = photo.get("same_area_visible_in_both")
    if _is_yes(comparable) and _is_yes(same_area):
        return "Verified (the same area is visible in both move-in and move-out photos)."
    if _is_yes(comparable):
        return "Not confirmed (the same area is not confirmed in both photos)."
    if isinstance(comparable, str) and comparable.strip().lower() == "no":
        return "Not verified (the photos were reported as not comparable)."
    if same_area is False or (
        isinstance(same_area, str) and same_area.strip().lower() == "false"
    ):
        return "Not verified (the same area is not visible in both photos)."
    return "Not confirmed (the comparison does not confirm a matching area)."


def _section_heading(text: str, styles):
    return KeepTogether([
        Paragraph(_text(text), styles["SectionHeading"]),
        HRFlowable(
            width="100%",
            thickness=0.6,
            color=_BORDER,
            spaceBefore=0,
            spaceAfter=5,
        ),
    ])


def _banded_table(rows, col_widths):
    table = Table(rows, colWidths=col_widths, repeatRows=1, splitByRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _MUTED_ROW]),
        ("GRID", (0, 0), (-1, -1), 0.35, _BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _append_repair_benchmark(
    story,
    benchmark,
    styles,
    heading="Repair Price Intelligence Benchmark",
) -> None:
    if not benchmark:
        return

    story.append(_section_heading(heading, styles))
    if benchmark.get("status") != "matched":
        message = benchmark.get("explanation")
        if not message:
            message = (
                "The deduction amount was unavailable for comparison."
                if benchmark.get("status") == "no_amount"
                else "No matching sample-rate category was available for this deduction at the selected locality."
            )
        story.append(Paragraph(_text(message), styles["ReportBody"]))
        return

    city = str(benchmark.get("city") or "").strip()
    locality = str(benchmark.get("locality") or "").strip()
    place = ", ".join(part for part in (city, locality) if part) or "the selected locality"
    category = str(benchmark.get("category") or "local repair").strip()
    lower = float(benchmark.get("lower") or 0)
    upper = float(benchmark.get("upper") or 0)
    claimed_amount = float(benchmark.get("claimed_amount") or 0)
    unit_suffix = f" per {_text(benchmark['unit'])}" if benchmark.get("unit") else ""
    range_text = f"{_money(lower)} to {_money(upper)}{unit_suffix}"
    position = benchmark.get("position")
    if position == "below":
        difference = benchmark.get("difference_percent")
        comparison = (
            f"is {float(difference):.1f}% below"
            if difference is not None
            else "is below"
        )
    elif position == "above":
        difference = benchmark.get("difference_percent")
        comparison = (
            f"is {float(difference):.1f}% above"
            if difference is not None
            else "is above"
        )
    elif position == "within":
        comparison = "falls within"
    else:
        comparison = "was compared with"

    count = int(benchmark.get("record_count") or 0)
    record_text = (
        f"based on {count} matched sample record{'s' if count != 1 else ''}"
        if count
        else "based on the available matched samples"
    )
    damage_cause = str(benchmark.get("damage_cause") or "").strip()
    claim_subject = (
        f"For {_text(damage_cause)}, the claimed amount"
        if damage_cause
        else "The claimed amount"
    )
    story.append(Paragraph(
        f"{claim_subject} of {_money(claimed_amount)} {comparison} the typical "
        f"{_text(place)} range for {_text(category)} of {range_text} "
        f"({record_text}).",
        styles["ReportBody"],
    ))

    records = benchmark.get("matched_records") or []
    if records:
        rows = [[
            Paragraph("City / locality", styles["TableHeader"]),
            Paragraph("Category / item", styles["TableHeader"]),
            Paragraph("Local sample range", styles["TableHeader"]),
        ]]
        for row in records[:8]:
            category_or_item = (
                row.get("category")
                or benchmark.get("category")
                or row.get("item", "")
            )
            sample_range = (
                f"{_money(row.get('minimum'))} to {_money(row.get('maximum'))}"
                + (f" per {_text(row['unit'])}" if row.get("unit") else "")
            )
            rows.append([
                Paragraph(
                    _text(f"{row.get('city', '')}, {row.get('locality', '')}"),
                    styles["TableCell"],
                ),
                Paragraph(_text(category_or_item), styles["TableCell"]),
                Paragraph(sample_range, styles["TableCell"]),
            ])
        story.append(KeepTogether([
            _banded_table(rows, [2.0 * inch, 2.0 * inch, 3.3 * inch]),
            Spacer(1, 5),
        ]))

    story.append(Paragraph(
        _text(benchmark.get(
            "sample_data_warning",
            "Sample data only; not a verified quotation or determination of actual cost.",
        )),
        styles["SmallText"],
    ))
    if benchmark.get("llm_explanation"):
        story.append(Paragraph(
            "<b>Context:</b> " + _text(benchmark["llm_explanation"])
            + " This explanation does not change the calculation or audit classification.",
            styles["SmallText"],
        ))


def _append_photo_comparison(story, photo, styles, title=None, finding=None):
    if title:
        story.append(_section_heading(title, styles))
    if photo.get("mode") != "llm_analyzed":
        story.append(Paragraph(
            "A linked photo pair is available, but no completed comparison is stored. "
            + _text(photo.get("error") or photo.get("limitations") or "No analysis details are available."),
            styles["ReportBody"],
        ))
        return

    if photo.get("observation"):
        story.append(Paragraph(
            "<b>Summary:</b> " + _text(photo["observation"])[:1500],
            styles["ReportBody"],
        ))

    story.append(Paragraph(
        "<b>Visual Match:</b> " + _text(_photo_match_text(photo)),
        styles["ReportBody"],
    ))

    for key, label in (
        ("move_in_condition", "Move-in condition"),
        ("move_out_condition", "Move-out condition"),
    ):
        if photo.get(key):
            story.append(Paragraph(
                f"<b>{label}:</b> {_text(photo[key])[:1000]}",
                styles["ReportBody"],
            ))

    changes = photo.get("observed_changes") or []
    if changes:
        rows = [[
            Paragraph("Observed feature / location", styles["TableHeader"]),
            Paragraph("Move-in condition", styles["TableHeader"]),
            Paragraph("Move-out condition", styles["TableHeader"]),
            Paragraph("Interpretation", styles["TableHeader"]),
        ]]
        for change in changes[:8]:
            feature = change.get("feature") or "Visible feature"
            if change.get("location"):
                feature += f" ({change['location']})"
            rows.append([
                Paragraph(_text(feature)[:400], styles["TableCell"]),
                Paragraph(
                    _text(change.get("move_in_condition") or "Not described")[:400],
                    styles["TableCell"],
                ),
                Paragraph(
                    _text(change.get("move_out_condition") or "Not described")[:400],
                    styles["TableCell"],
                ),
                Paragraph(
                    _text(change.get("interpretation") or "No interpretation provided")[:500],
                    styles["TableCell"],
                ),
            ])
        comparison_table = _banded_table(
            rows,
            [1.35 * inch, 1.55 * inch, 1.55 * inch, 2.85 * inch],
        )
        story.append(KeepTogether([
            _section_heading("Move-In vs. Move-Out Comparison", styles),
            comparison_table,
            Spacer(1, 5),
        ]))
    elif photo.get("visible_differences"):
        story.append(Paragraph(
            "<b>Observed changes:</b> "
            + _text("; ".join(photo["visible_differences"]))[:1200],
            styles["ReportBody"],
        ))
    else:
        story.append(Paragraph(
            "<b>Observed changes:</b> No specific visible differences were reported.",
            styles["ReportBody"],
        ))

    uncertainty = photo.get("model_uncertainty")
    if uncertainty:
        story.append(Paragraph(
            "<b>Comparison uncertainty:</b> "
            + _text(uncertainty)
            + " (qualitative, not a calibrated confidence score).",
            styles["SmallText"],
        ))
    if photo.get("possible_damage_or_deterioration"):
        story.append(Paragraph(
            "<b>Possible damage (not a finding of responsibility):</b> "
            + _text("; ".join(photo["possible_damage_or_deterioration"]))[:1200],
            styles["SmallText"],
        ))
    if photo.get("not_reliably_comparable"):
        story.append(Paragraph(
            "<b>Comparison limitations:</b> "
            + _text("; ".join(photo["not_reliably_comparable"]))[:1000],
            styles["SmallText"],
        ))
    if photo.get("limitations"):
        story.append(Paragraph(
            "<b>Model-stated limitations:</b> " + _text(photo["limitations"])[:1000],
            styles["SmallText"],
        ))
    if finding:
        story.append(Paragraph(
            f"<b>Relevance to deduction:</b> This stored photo comparison concerns "
            f"{_text(finding.get('photo_item_name') or 'the linked photo item')}; "
            "it was not a deduction assessment. It may provide visual context for "
            f"{_text(finding.get('description', 'the claim'))}, but does not establish "
            "that the claimed repair was required or its cost.",
            styles["SmallText"],
        ))
    story.append(Paragraph(
        "Photographs can show visible differences, but cannot independently establish when a change "
        "occurred, who caused it, actual repair cost, or legal responsibility.",
        styles["SmallText"],
    ))


def generate_audit_pdf(case, findings, totals, reports_dir: Path, photo_pairs=None) -> str:
    case = {key: case[key] for key in case.keys()}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"depositproof_audit_{case['id']}_{uuid.uuid4().hex[:6]}.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.3,
        textColor=colors.HexColor("#334155"),
        wordWrap="CJK",
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["ReportBody"],
        fontSize=8.5,
        leading=11.5,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14.5,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["SmallText"],
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["SmallText"],
        fontSize=8.2,
        leading=10.6,
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="CardTitle",
        parent=styles["ReportBody"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="CardMeta",
        parent=styles["SmallText"],
        fontSize=8,
        leading=10.5,
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="StatusBadge",
        parent=styles["SmallText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="ClauseHeader",
        parent=styles["SmallText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        name="ClauseText",
        parent=styles["ReportBody"],
        fontSize=9,
        leading=12.4,
        spaceAfter=0,
    ))

    claimed_total = float(case.get("claimed_deductions") or 0)
    itemized_total = sum(float(finding.get("amount") or 0) for finding in findings)

    story = [
        Paragraph("DepositProof AI", styles["ReportTitle"]),
        Paragraph("Rental Security Deposit Evidence Audit", styles["ReportSubtitle"]),
        Paragraph(
            f"<b>Case:</b> {_text(case.get('name', ''))}"
            f" &nbsp;&nbsp; <b>Case ID:</b> {_text(case.get('id', ''))}",
            styles["ReportBody"],
        ),
        Paragraph(
            "<b>Recommended refund basis:</b> Security deposit less verified deductions, "
            "floored at zero.",
            styles["SmallText"],
        ),
    ]
    proposed_refund = case.get("proposed_refund")
    if proposed_refund is not None:
        story.append(Paragraph(
            f"<b>Proposed refund:</b> {_money(proposed_refund)}",
            styles["SmallText"],
        ))

    difference = claimed_total - itemized_total
    if difference > 0.005:
        story.append(Paragraph(
            f"<b>Unitemized claimed balance:</b> {_money(difference)} of the landlord's "
            "claimed deductions is not itemized in the findings below.",
            styles["ReportBody"],
        ))
    elif difference < -0.005:
        story.append(Paragraph(
            f"<b>Reconciliation:</b> Itemized deductions exceed the landlord's stated "
            f"total by {_money(abs(difference))}.",
            styles["ReportBody"],
        ))

    story.append(_section_heading("Deduction Breakdown", styles))
    for i, finding in enumerate(findings, 1):
        classification = finding.get("classification", "Insufficiently Documented")
        status_background, status_foreground = _STATUS_COLORS.get(
            classification,
            _STATUS_COLORS["Insufficiently Documented"],
        )
        title = f"{i:02d} - {_title_case(finding.get('description', 'Deduction'))}"
        status_style = ParagraphStyle(
            name=f"StatusBadge{i}",
            parent=styles["StatusBadge"],
            textColor=status_foreground,
        )
        card = Table(
            [[
                Paragraph(_text(title), styles["CardTitle"]),
                Paragraph(
                    f"<b>Claimed</b><br/>{_money(finding.get('amount'))}",
                    styles["CardMeta"],
                ),
                Paragraph(_text(classification), status_style),
            ]],
            colWidths=[3.35 * inch, 1.45 * inch, 2.5 * inch],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.white),
            ("BACKGROUND", (2, 0), (2, 0), status_background),
            ("BOX", (0, 0), (-1, -1), 0.7, _BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether([card, Spacer(1, 6)]))

        explanation = str(finding.get("explanation") or "No explanation available.")
        story.append(_section_heading("Executive Takeaway", styles))
        for paragraph in re.split(r"\n\s*\n", explanation.strip()):
            if paragraph.strip():
                story.append(Paragraph(
                    _text(paragraph).replace("\n", "<br/>"),
                    styles["ReportBody"],
                ))

        clauses = finding.get("clauses") or []
        if clauses:
            story.append(_section_heading("Relevant Agreement Clauses", styles))
            for clause_number, clause in enumerate(clauses[:5], 1):
                verification = (
                    "Clause Verified" if clause.get("quote_verified")
                    else "Clause Not Verified"
                )
                header = (
                    f"{clause_number:02d}_{_text(clause.get('filename', 'Agreement'))}"
                    f" | Page {_text(clause.get('page', 'unknown'))} | {verification}"
                )
                quote = _text(clause.get("quote", "")).replace("\n", "<br/>")
                callout = Table(
                    [
                        [Paragraph(header, styles["ClauseHeader"])],
                        [Paragraph(quote, styles["ClauseText"])],
                    ],
                    colWidths=[doc.width],
                    repeatRows=1,
                    splitByRow=1,
                    splitInRow=1,
                )
                callout.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), _MUTED_ROW),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#2563EB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]))
                story.append(KeepTogether([callout, Spacer(1, 6)]))

        story.append(_section_heading("Limitations", styles))
        story.append(Paragraph(
            _text(_report_limitations(finding.get("limitations"))),
            styles["ReportBody"],
        ))

        photo = finding.get("photo_observation") or {}
        if photo:
            _append_photo_comparison(
                story,
                photo,
                styles,
                title="Photo Comparison",
                finding=finding,
            )
        else:
            story.append(Paragraph(
                "No linked usable photo observation is available for this deduction.",
                styles["SmallText"],
            ))

        _append_repair_benchmark(story, finding.get("repair_price_benchmark"), styles)
        _append_repair_benchmark(
            story,
            finding.get("locality_benchmark"),
            styles,
            heading="Locality Benchmark",
        )
        if finding.get("ai_error"):
            story.append(Paragraph(
                f"<b>Hosted AI unavailable:</b> {_text(finding['ai_error'])[:1000]} "
                "(the evidence review was used as a fallback.)",
                styles["SmallText"],
            ))
        story.append(Spacer(1, 8))

    if photo_pairs:
        story.append(_section_heading(
            "Saved Photo Comparisons Not Linked to a Deduction",
            styles,
        ))
        story.append(Paragraph(
            "These stored photo comparisons are not associated with a specific deduction "
            "and should not be treated as evidence for a particular claim.",
            styles["ReportBody"],
        ))
        for pair in photo_pairs:
            photo = json.loads(pair["observation_json"]) if pair["observation_json"] else {
                "mode": "not_analyzed",
                "limitations": "No photo observation is stored.",
            }
            _append_photo_comparison(
                story,
                photo,
                styles,
                title=_title_case(pair["item_name"]),
            )

    story += [
        Spacer(1, 8),
        _section_heading("Disclaimer", styles),
        Paragraph(
            "This report summarizes evidence classifications and includes hosted-model "
            "explanations or photo comparisons only where explicitly indicated. It is not "
            "legal advice, a legal judgment, or a guarantee of refund. Missing documentation "
            "does not by itself prove a claim is invalid. Review original documents and seek "
            "qualified advice where appropriate.",
            styles["ReportBody"],
        ),
    ]
    doc.build(story)
    return str(path)
