import unittest
from unittest.mock import patch

import evidence_engine as ee


class EvidenceEngineTests(unittest.TestCase):
    def test_verified_matching_evidence_can_be_supported(self):
        payload = {
            "description": "Repainting the living room",
            "category": "Painting / repainting",
            "amount": 2500,
            "clauses": [{
                "quote": "Tenant must pay for repainting due to damage.",
                "source_text": "Tenant must pay for repainting due to damage.",
                "quote_verified": True,
                "relevance": "relevant",
                "filename": "agreement.pdf",
                "page": "2",
            }],
            "invoice": {
                "filename": "paint.pdf",
                "extracted_text": "Painting work, total 2500",
                "service_description": "Painting work",
                "total_amount": 2500,
            },
            "photo_observation": {
                "mode": "llm_analyzed",
                "comparable": "yes",
                "visible_differences": ["A new paint mark is visible."],
            },
        }

        result = ee.classify_deduction(payload)

        self.assertEqual(result["classification"], ee.SUPPORTED)

    def test_invoice_total_mismatch_is_inconsistent(self):
        payload = {
            "description": "Repainting the living room",
            "category": "Painting",
            "amount": 2500,
            "invoice": {
                "filename": "paint.pdf",
                "extracted_text": "Painting work, total 1800",
                "service_description": "Painting work",
                "total_amount": 1800,
            },
        }

        result = ee.classify_deduction(payload)

        self.assertEqual(result["classification"], ee.INCONSISTENT)
        self.assertEqual(result["amount_status"], "mismatch")

    def test_missing_evidence_remains_insufficient(self):
        result = ee.classify_deduction({
            "description": "Repainting the living room",
            "category": "Painting",
            "amount": 2500,
        })

        self.assertEqual(result["classification"], ee.INSUFFICIENT)

    def test_fabricated_quote_is_rejected(self):
        self.assertFalse(ee.verify_quote(
            "Tenant must replace all furniture.", "Tenant will keep the unit clean."
        ))

    def test_keyword_retrieval_returns_a_verifiable_source_passage(self):
        source = (
            "The tenant must keep the premises clean.\n\n"
            "The tenant must pay for repainting caused by damage beyond ordinary wear."
        )
        with patch.object(ee, "_load_embedder", return_value=None):
            passages = ee.find_relevant_clauses(
                "repainting walls", source, "agreement.pdf", {"1": source},
                category="Painting",
            )

        self.assertTrue(passages)
        self.assertTrue(any(p["quote_verified"] for p in passages))
        self.assertTrue(all(p["retrieval_method"] == "keyword_fallback" for p in passages))
        self.assertIn(passages[0]["quote"], source)

    def test_embedding_retrieval_keeps_quotes_from_the_source_text(self):
        class FakeEmbedder:
            def encode(self, texts, normalize_embeddings, show_progress_bar):
                return [[1.0, 0.0] for _ in texts]

        source = "Tenant must pay for repainting caused by damage beyond ordinary wear."
        with patch.object(ee, "_load_embedder", return_value=FakeEmbedder()):
            passages = ee.find_relevant_clauses(
                "repainting walls", source, "agreement.pdf", {"1": source},
                category="Painting",
            )

        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["retrieval_method"], f"semantic:{ee.DEFAULT_EMBEDDING_MODEL}")
        self.assertEqual(passages[0]["quote"], source)
        self.assertTrue(passages[0]["quote_verified"])


if __name__ == "__main__":
    unittest.main()
