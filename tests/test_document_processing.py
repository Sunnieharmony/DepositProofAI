import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_processing import extract_invoice_fields, extract_pdf
from reportlab.pdfgen import canvas


class DocumentProcessingTests(unittest.TestCase):
    def _create_pdf(self, path: Path, pages: list[str]) -> None:
        pdf = canvas.Canvas(str(path))
        for text in pages:
            if text:
                pdf.drawString(72, 720, text)
            pdf.showPage()
        pdf.save()

    def _force_pymupdf_import_failure(self):
        original_import = builtins.__import__

        def blocked_pdf_backend(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("native backend blocked")
            return original_import(name, *args, **kwargs)

        return patch("builtins.__import__", side_effect=blocked_pdf_backend)

    def test_pypdf_fallback_extracts_text_and_preserves_page_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "agreement.pdf"
            self._create_pdf(path, ["Move-in inspection clause.", "Tenant must pay for damage."])

            with self._force_pymupdf_import_failure():
                text, pages, status = extract_pdf(str(path))

        self.assertIn("Move-in inspection clause.", text)
        self.assertIn("Tenant must pay for damage.", text)
        self.assertIn("Move-in inspection clause.", pages["1"])
        self.assertIn("Tenant must pay for damage.", pages["2"])
        self.assertEqual(status, "extracted_with_pypdf")

    def test_pypdf_fallback_reports_pdf_with_no_extractable_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scanned.pdf"
            self._create_pdf(path, ["", ""])

            with self._force_pymupdf_import_failure():
                text, pages, status = extract_pdf(str(path))

        self.assertEqual(text, "")
        self.assertEqual(pages, {"1": "", "2": ""})
        self.assertIn("no_text_found", status)
        self.assertIn("scanned PDF", status)
        self.assertIn("OCR is not implemented", status)

    def test_invoice_total_and_date_are_extracted_from_source_text(self):
        fields = extract_invoice_fields(
            "Invoice\nVendor: Paint Co\nInvoice Date: 15/01/2025\n"
            "Description: Wall repainting\nGrand Total: INR 2,500"
        )

        self.assertEqual(fields["total_amount"], 2500)
        self.assertEqual(fields["invoice_date"], "2025-01-15")
        self.assertIn("repainting", fields["service_description"].lower())


if __name__ == "__main__":
    unittest.main()
