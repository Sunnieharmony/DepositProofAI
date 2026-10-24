import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from pypdf import PdfReader


class AppWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.original_paths = (app.DB_PATH, app.UPLOADS, app.REPORTS)
        self.addCleanup(self._restore_paths)
        app.DB_PATH = root / "case.sqlite3"
        app.UPLOADS = root / "uploads"
        app.REPORTS = root / "reports"
        app.UPLOADS.mkdir()
        app.REPORTS.mkdir()
        app.init_db()
        self.case_id = "TESTCASE01"
        with app.db() as con:
            con.execute(
                """INSERT INTO cases(id,name,deposit,claimed_deductions,proposed_refund,
                   move_in,move_out,currency,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (self.case_id, "Test property", 10000, 500, 9500,
                 "2025-01-01", "2025-12-31", "INR", "2025-01-01T00:00:00"),
            )

    def _restore_paths(self):
        app.DB_PATH, app.UPLOADS, app.REPORTS = self.original_paths

    def _add_deduction(self, deduction_id, description, photo_pair_id=None):
        with app.db() as con:
            con.execute(
                """INSERT INTO deductions(id,case_id,category,description,amount,photo_pair_id,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (deduction_id, self.case_id, "Painting / repainting", description, 250,
                 photo_pair_id, "2025-01-01T00:00:00"),
            )

    def test_welcome_action_opens_case_step_and_hides_welcome(self):
        welcome, application, *step_state = app._enter_application()

        self.assertFalse(welcome["visible"])
        self.assertTrue(application["visible"])
        self.assertEqual(len(step_state), 6)
        self.assertEqual([item["visible"] for item in step_state[:4]], [True, False, False, False])
        self.assertIn("Audit case", step_state[4])
        self.assertIn('aria-current="step"', step_state[4])
        self.assertEqual(step_state[5], "")

    def test_mobile_navigation_returns_all_workflow_outputs(self):
        state = app._navigate_to_step("Evidence")

        self.assertEqual(len(state), 6)
        self.assertEqual([item["visible"] for item in state[:4]], [False, True, False, False])
        self.assertIn("Evidence", state[4])
        self.assertEqual(state[5], "")

    def test_case_continue_requires_saved_case_and_preserves_case_state(self):
        missing_state = app._continue_from_case(None)
        self.assertEqual([item["visible"] for item in missing_state[:4]], [True, False, False, False])
        self.assertIn("Create or select an audit case", missing_state[5])

        with app.db() as con:
            con.execute(
                """INSERT INTO cases(id,name,deposit,claimed_deductions,proposed_refund,
                   move_in,move_out,currency,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                ("NAVCASE01", "Navigation property", 12000, 2000, 10000,
                 "2025-01-01", "2025-12-31", "INR", "2025-01-02"),
            )
        state = app._continue_from_case("NAVCASE01")

        self.assertEqual([item["visible"] for item in state[:4]], [False, True, False, False])
        with app.db() as con:
            saved = con.execute(
                "SELECT name,deposit,claimed_deductions FROM cases WHERE id=?",
                ("NAVCASE01",),
            ).fetchone()
        self.assertEqual(
            (saved["name"], saved["deposit"], saved["claimed_deductions"]),
            ("Navigation property", 12000, 2000),
        )
        findings_state = app._continue_from_evidence("NAVCASE01")
        self.assertEqual(
            [item["visible"] for item in findings_state[:4]],
            [False, False, True, False],
        )

    def test_final_report_requires_completed_audit_output(self):
        blocked_state = app._continue_from_review("")
        self.assertEqual([item["visible"] for item in blocked_state[:4]], [False, False, True, False])
        self.assertIn("Run the evidence audit", blocked_state[5])

        report_state = app._continue_from_review("# Audit report")
        self.assertEqual(
            [item["visible"] for item in report_state[:4]],
            [False, False, False, True],
        )

    def test_photo_pair_can_be_linked_to_its_deduction(self):
        root = Path(self.temp_dir.name)
        move_in = root / "move-in.png"
        move_out = root / "move-out.png"
        move_in.write_bytes(b"move-in test fixture")
        move_out.write_bytes(b"move-out test fixture")
        observation = {
            "observation": "A mark appears in the move-out image.",
            "visible_differences": ["A mark is visible."],
            "possible_damage_or_deterioration": [],
            "comparable": "yes",
            "mode": "llm_analyzed",
        }
        with patch.object(app, "validate_image_file", return_value=(True, "OK")), \
                patch.object(app, "llm_is_configured", return_value=True), \
                patch.object(app, "analyze_image_pair_with_llm", return_value=observation):
            upload_result, _ = app.upload_photo_pair(
                self.case_id, "Living room wall", str(move_in), str(move_out)
            )

        self.assertIn("Vision-model comparison completed", upload_result)
        self.assertIn("A mark appears in the move-out image.", upload_result)
        self.assertIn("A mark is visible.", upload_result)
        with app.db() as con:
            pair = con.execute("SELECT id,status FROM photo_pairs WHERE case_id=?", (self.case_id,)).fetchone()
        self.assertEqual(pair["status"], "llm_analyzed")
        photo_choice = f"{pair['id']} — Living room wall (llm_analyzed)"
        app.add_deduction(
            self.case_id, "Painting / repainting", "Repainting living room", 500,
            "None", photo_choice,
        )
        with app.db() as con:
            deduction = con.execute(
                "SELECT photo_pair_id FROM deductions WHERE case_id=?", (self.case_id,)
            ).fetchone()
        self.assertEqual(deduction["photo_pair_id"], pair["id"])

    def test_photo_upload_shows_clear_error_when_analysis_fails(self):
        root = Path(self.temp_dir.name)
        move_in = root / "move-in-failed.png"
        move_out = root / "move-out-failed.png"
        move_in.write_bytes(b"move-in test fixture")
        move_out.write_bytes(b"move-out test fixture")
        with patch.object(app, "validate_image_file", return_value=(True, "OK")), \
                patch.object(app, "llm_is_configured", return_value=True), \
                patch.object(app, "analyze_image_pair_with_llm", return_value=None), \
                patch.object(app, "get_last_error", return_value="Gemini quota limit reached"):
            upload_result, _ = app.upload_photo_pair(
                self.case_id, "Bedroom wall", str(move_in), str(move_out)
            )

        self.assertIn("Automated image comparison was not completed.", upload_result)
        self.assertIn("Gemini quota limit reached", upload_result)

    def test_rerunning_audit_replaces_findings_instead_of_duplicating_them(self):
        with app.db() as con:
            con.execute(
                """INSERT INTO deductions(id,case_id,category,description,amount,created_at)
                   VALUES(?,?,?,?,?,?)""",
                ("DEDUCTION01", self.case_id, "Cleaning", "Move-out cleaning <wall>", 500,
                 "2025-01-01T00:00:00"),
            )
        with patch.object(app, "llm_is_configured", return_value=False):
            app.run_audit(self.case_id)
            app.run_audit(self.case_id)

        with app.db() as con:
            findings = con.execute(
                "SELECT finding_json FROM findings WHERE case_id=? AND deduction_id=?",
                (self.case_id, "DEDUCTION01"),
            ).fetchall()
        self.assertEqual(len(findings), 1)
        self.assertEqual(
            json.loads(findings[0]["finding_json"])["classification"],
            "Insufficiently Documented",
        )

    def test_saved_price_benchmark_is_reported_without_changing_classification(self):
        with app.db() as con:
            con.execute(
                """INSERT INTO deductions(id,case_id,category,description,amount,created_at)
                   VALUES(?,?,?,?,?,?)""",
                ("BENCHMARK01", self.case_id, "Painting / repainting",
                 "Repainting a 1BHK", 10000, "2025-01-01T00:00:00"),
            )
        output = app.run_repair_price_comparison(
            self.case_id,
            "BENCHMARK01 — Repainting a 1BHK (₹10,000)",
            "Painting / Repainting",
            10000,
            "Bengaluru",
            "HSR Layout",
            False,
        )
        self.assertIn("Saved to this deduction", output)
        self.assertIn("8,800", output)
        self.assertIn("15,400", output)

        rules_result = {
            "classification": "Inconsistent",
            "explanation": "The rules identified a contradiction in the evidence.",
        }
        with patch.object(app, "llm_is_configured", return_value=False), \
                patch.object(app, "classify_deduction", return_value=rules_result):
            markdown, pdf_path, _ = app.run_audit(self.case_id)
            app.run_audit(self.case_id)

        self.assertIn("Repair-price benchmark (context only; does not change the classification)", markdown)
        self.assertIn("Sample repair-rate data only", markdown)
        self.assertTrue(Path(pdf_path).is_file())
        pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
        self.assertIn("Repair-price benchmark", pdf_text)
        self.assertIn("Sample repair-rate data only", pdf_text)
        with app.db() as con:
            finding = con.execute(
                "SELECT finding_json FROM findings WHERE deduction_id=?",
                ("BENCHMARK01",),
            ).fetchone()
            benchmark_count = con.execute(
                "SELECT COUNT(*) FROM repair_benchmarks WHERE deduction_id=?",
                ("BENCHMARK01",),
            ).fetchone()[0]
        saved_finding = json.loads(finding["finding_json"])
        self.assertEqual(saved_finding["classification"], "Inconsistent")
        self.assertEqual(saved_finding["repair_price_benchmark"]["position"], "within")
        self.assertEqual(benchmark_count, 1)

    def test_gemini_benchmark_failure_keeps_deterministic_result(self):
        with patch.object(app, "explain_repair_benchmark_with_llm", return_value=None), \
                patch.object(app, "get_last_error", return_value="Gemini quota exhausted"):
            output = app.run_repair_price_comparison(
                None, "None", "Painting / Repainting", 16000,
                "Bengaluru", "HSR Layout", True,
            )

        self.assertIn("Deterministic explanation", output)
        self.assertIn("Hosted-model explanation was not used", output)
        self.assertIn("Gemini quota exhausted", output)
        self.assertIn("above the matched sample range", output)

    def test_claim_amount_mismatch_does_not_attach_benchmark_to_deduction(self):
        with app.db() as con:
            con.execute(
                """INSERT INTO deductions(id,case_id,category,description,amount,created_at)
                   VALUES(?,?,?,?,?,?)""",
                ("BENCHMARK02", self.case_id, "Painting / repainting",
                 "Repainting a 1BHK", 10000, "2025-01-01T00:00:00"),
            )
        output = app.run_repair_price_comparison(
            self.case_id,
            "BENCHMARK02 — Repainting a 1BHK (₹10,000)",
            "Painting / Repainting",
            12000,
            "Bengaluru",
            "HSR Layout",
            False,
        )

        self.assertIn("does not match the saved deduction amount", output)
        with app.db() as con:
            count = con.execute(
                "SELECT COUNT(*) FROM repair_benchmarks WHERE deduction_id=?",
                ("BENCHMARK02",),
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_audit_verifies_retrieved_quotes_against_the_agreement_source(self):
        agreement_text = "Tenant must pay for repainting due to tenant-caused damage beyond ordinary wear."
        with app.db() as con:
            con.execute(
                """INSERT INTO documents(id,case_id,filename,category,stored_path,extracted_text,
                   page_text_json,status,invoice_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                ("AGREEMENT01", self.case_id, "agreement.pdf", "Rental agreement", "agreement.pdf",
                 agreement_text, json.dumps({"1": agreement_text}), "extracted", "{}", "2025-01-01"),
            )
            con.execute(
                """INSERT INTO documents(id,case_id,filename,category,stored_path,extracted_text,
                   page_text_json,status,invoice_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                ("INVOICE0001", self.case_id, "painting.pdf", "Invoice / receipt", "painting.pdf",
                 "Painting work total 500", "{}", "extracted",
                 json.dumps({"total_amount": 500, "service_description": "Painting work"}),
                 "2025-01-01"),
            )
            con.execute(
                """INSERT INTO photo_pairs(id,case_id,item_name,observation_json,status,created_at)
                   VALUES(?,?,?,?,?,?)""",
                ("PHOTO000001", self.case_id, "Living room wall",
                 json.dumps({
                     "observation": "A visible paint mark is present.",
                     "visible_differences": ["A paint mark is present."],
                     "comparable": "yes",
                     "mode": "llm_analyzed",
                 }),
                 "llm_analyzed", "2025-01-01"),
            )
            con.execute(
                """INSERT INTO deductions(id,case_id,category,description,amount,invoice_document_id,
                   photo_pair_id,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                ("DEDUCTION02", self.case_id, "Painting / repainting",
                 "Repainting living room wall", 500, "INVOICE0001", "PHOTO000001", "2025-01-01"),
            )

        with patch.object(app, "llm_is_configured", return_value=False):
            app.run_audit(self.case_id)

        with app.db() as con:
            finding = con.execute(
                "SELECT finding_json FROM findings WHERE deduction_id=?",
                ("DEDUCTION02",),
            ).fetchone()
        result = json.loads(finding["finding_json"])
        self.assertEqual(result["classification"], "Supported")
        self.assertTrue(result["clauses"][0]["quote_verified"])
        self.assertEqual(result["clauses"][0]["page"], "1")

    def test_mixed_assessment_modes_and_photo_observations_are_reported_per_deduction(self):
        observation = {
            "mode": "llm_analyzed",
            "observation": "A small mark is visible in the later image.",
            "move_in_condition": "The wall appears clear in the photographed area.",
            "move_out_condition": "A small dark mark appears in the photographed area.",
            "observed_changes": [{
                "feature": "Dark mark",
                "move_in_condition": "No mark is apparent.",
                "move_out_condition": "A dark mark is visible.",
                "location": "Near the baseboard",
                "interpretation": "A visible difference; its cause is not established.",
            }],
            "visible_differences": ["A small dark mark is visible."],
            "possible_damage_or_deterioration": [],
            "comparable": "yes",
            "same_area_visible_in_both": True,
            "model_uncertainty": "medium",
            "limitations": "Lighting differs slightly.",
        }
        with app.db() as con:
            con.execute(
                """INSERT INTO photo_pairs(id,case_id,item_name,observation_json,status,created_at)
                   VALUES(?,?,?,?,?,?)""",
                ("PHOTO000002", self.case_id, "Living room wall", json.dumps(observation),
                 "llm_analyzed", "2025-01-01"),
            )
        self._add_deduction("LLMDEDUCTION01", "Repainting living room wall", "PHOTO000002")
        self._add_deduction("RULEDEDUCTION01", "Repainting hallway")

        llm_assessment = {
            "classification": "Supported",
            "rules_classification": "Supported",
            "llm_classification_suggestion": "Supported",
            "explanation": "The rule checks support this claim based on the provided evidence.",
            "explanation_source": "model",
            "limitations": "Model explanation is limited to the provided facts.",
        }
        fallback_assessment = {
            "classification": "Insufficiently Documented",
            "explanation": "The available evidence does not establish the claim.",
            "evidence_reviewed": "No verified evidence was linked.",
            "missing_or_conflicting_evidence": "Missing invoice and agreement clause.",
            "limitations": "Rules engine limitation.",
        }
        with patch.object(app, "llm_is_configured", return_value=True), \
                patch.object(app, "find_relevant_clauses", return_value=[]), \
                patch.object(app, "assess_deduction_with_llm", side_effect=[llm_assessment, None]), \
                patch.object(app, "get_last_error", return_value="Gemini quota exhausted"), \
                patch.object(app, "classify_deduction", return_value=fallback_assessment):
            markdown, pdf_path, _ = app.run_audit(self.case_id)

        self.assertIn(
            "**Assessment modes by deduction:** 1 LLM explanation with rules-engine classification; "
            "1 rules-based fallback after LLM failure",
            markdown,
        )
        self.assertNotIn("**Mode:** LLM-assisted", markdown)
        self.assertIn("Validated LLM explanation; classification determined by evidence rules", markdown)
        self.assertIn("**LLM suggestion (not authoritative):** Supported", markdown)
        self.assertIn("Rules-based fallback after hosted LLM assessment failed", markdown)
        self.assertIn("Gemini quota exhausted", markdown)
        self.assertIn("| Observed feature / location | Move-in condition | Move-out condition | Interpretation |", markdown)
        self.assertIn("Near the baseboard", markdown)
        self.assertIn("Lighting differs slightly.", markdown)
        self.assertIn("this is not a calibrated confidence score.", markdown)
        self.assertIn("it was not a deduction assessment", markdown)
        self.assertIn("No linked usable photo observation is available", markdown)
        self.assertNotIn("Rules-based assessment of the uploaded evidence only.", markdown)

        pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
        self.assertIn("Assessment modes by deduction", pdf_text)
        self.assertIn("LLM explanation with rules-engine classification", pdf_text)
        self.assertIn("rules-based fallback after LLM failure", pdf_text)
        self.assertIn("Observed feature /", pdf_text)
        self.assertIn("location", pdf_text)
        self.assertIn("Near the", pdf_text)
        self.assertIn("baseboard", pdf_text)
        self.assertIn("No linked usable photo observation", pdf_text)

    def test_unlinked_photo_conclusion_is_included_without_associating_it_to_claim(self):
        observation = {
            "mode": "llm_analyzed",
            "observation": "A chipped area is visible in the move-out photograph.",
            "move_in_condition": "The photographed tile appears intact.",
            "move_out_condition": "A chipped edge is visible on the tile.",
            "observed_changes": [{
                "feature": "Chipped tile edge",
                "move_in_condition": "No chip is apparent.",
                "move_out_condition": "A small chip is visible.",
                "location": "Lower-right edge",
                "interpretation": "Visible difference only; cause is not established.",
            }],
            "visible_differences": ["A chip is visible on the tile."],
            "comparable": "yes",
            "same_area_visible_in_both": True,
            "model_uncertainty": "low",
            "limitations": "The tile is only partly visible.",
        }
        with app.db() as con:
            con.execute(
                """INSERT INTO photo_pairs(id,case_id,item_name,observation_json,status,created_at)
                   VALUES(?,?,?,?,?,?)""",
                ("PHOTO_UNLINKED", self.case_id, "Kitchen tile", json.dumps(observation),
                 "llm_analyzed", "2025-01-01"),
            )
        self._add_deduction("UNLINKED_DEDUCTION", "Kitchen repair")

        with patch.object(app, "llm_is_configured", return_value=False):
            markdown, pdf_path, _ = app.run_audit(self.case_id)

        self.assertIn("Saved photo comparisons not linked to a deduction", markdown)
        self.assertIn("Kitchen tile — status: llm_analyzed", markdown)
        self.assertIn("A chipped area is visible in the move-out photograph.", markdown)
        self.assertIn("Lower-right edge", markdown)
        self.assertIn("not linked to a deduction; do not treat it as evidence", markdown)

        with app.db() as con:
            finding = con.execute(
                "SELECT finding_json FROM findings WHERE deduction_id=?",
                ("UNLINKED_DEDUCTION",),
            ).fetchone()
        self.assertIsNone(json.loads(finding["finding_json"])["photo_observation"])

        pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
        self.assertIn("Saved photo comparisons not linked to a deduction", pdf_text)
        self.assertIn("A chipped area is visible in the move-out photograph.", pdf_text)
        self.assertIn("not associated with a specific deduction", pdf_text)


if __name__ == "__main__":
    unittest.main()
