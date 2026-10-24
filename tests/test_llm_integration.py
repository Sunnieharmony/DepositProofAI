import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

import llm_integration as llm


class LLMIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.token_config = patch.dict(os.environ, {
            "HF_TOKEN": "test-token-value",
            "LLM_PROVIDER": "huggingface",
        })
        self.token_config.start()
        self.addCleanup(self.token_config.stop)

    @staticmethod
    def _vision_response(**overrides):
        result = {
            "summary": "A visible mark appears in the second photograph.",
            "comparable": "yes",
            "same_area_visible_in_both": True,
            "visible_differences": ["A mark is visible."],
            "possible_damage_or_deterioration": [],
            "not_reliably_comparable": [],
            "uncertainty": "medium",
            "limitations": "Lighting differs slightly.",
        }
        result.update(overrides)
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    def test_both_prepared_images_reach_the_hosted_request(self):
        image_info = {"original_size": (100, 100), "sent_size": (100, 100), "sent_kb": 1, "format": "PNG"}
        with patch.object(llm, "prepare_image", side_effect=[
            ("data:image/jpeg;base64,movein", image_info),
            ("data:image/jpeg;base64,moveout", image_info),
        ]), patch.object(llm, "_post_chat", return_value=self._vision_response()) as request:
            result = llm.analyze_image_pair_with_llm("movein.png", "moveout.png", "living room wall")

        self.assertIsNotNone(result)
        self.assertEqual(result["images_sent"], 2)
        content = request.call_args.args[0]["messages"][0]["content"]
        sent_images = [part for part in content if part.get("type") == "image_url"]
        self.assertEqual(len(sent_images), 2)

    def test_real_valid_image_files_are_preprocessed_and_sent(self):
        with tempfile.TemporaryDirectory() as folder:
            move_in = Path(folder) / "move-in.png"
            move_out = Path(folder) / "move-out.png"
            Image.new("RGB", (32, 24), "white").save(move_in)
            Image.new("RGB", (32, 24), "gray").save(move_out)
            with patch.object(llm, "_post_chat", return_value=self._vision_response()) as request:
                result = llm.analyze_image_pair_with_llm(str(move_in), str(move_out), "wall")

        self.assertIsNotNone(result)
        self.assertEqual(result["images_sent"], 2)
        content = request.call_args.args[0]["messages"][0]["content"]
        image_urls = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
        self.assertEqual(len(image_urls), 2)
        self.assertTrue(all(url.startswith("data:image/jpeg;base64,") for url in image_urls))

    def test_malformed_vision_response_is_rejected_after_retry(self):
        image_info = {"original_size": (100, 100), "sent_size": (100, 100), "sent_kb": 1, "format": "PNG"}
        invalid = self._vision_response(same_area_visible_in_both="yes")
        with patch.object(llm, "prepare_image", return_value=("data:image/jpeg;base64,image", image_info)), \
                patch.object(llm, "_post_chat", side_effect=[invalid, invalid]) as request:
            result = llm.analyze_image_pair_with_llm("movein.png", "moveout.png", "wall")

        self.assertIsNone(result)
        self.assertEqual(request.call_count, 2)
        self.assertIn("same_area_visible_in_both", llm.get_last_error())

    def test_structured_vision_comparison_fields_are_validated_and_preserved(self):
        data = json.loads(self._vision_response()["choices"][0]["message"]["content"])
        data.update({
            "move_in_condition": "The wall appears clear.",
            "move_out_condition": "A small mark is visible.",
            "observed_changes": [{
                "feature": "Small mark",
                "move_in_condition": "No mark is apparent.",
                "move_out_condition": "A small mark is visible.",
                "location": "Near the baseboard",
                "interpretation": "A visible difference; cause unknown.",
            }],
        })

        validated = llm._validate_vision(data)

        self.assertEqual(validated["move_in_condition"], "The wall appears clear.")
        self.assertEqual(validated["move_out_condition"], "A small mark is visible.")
        self.assertEqual(validated["observed_changes"][0]["location"], "Near the baseboard")
        self.assertEqual(validated["uncertainty"], "medium")
        self.assertNotIn("confidence", validated)

    def test_malformed_structured_vision_fields_are_rejected(self):
        data = json.loads(self._vision_response()["choices"][0]["message"]["content"])
        data["observed_changes"] = [{"feature": "Mark", "location": 4}]

        with self.assertRaises(llm.LLMError):
            llm._validate_vision(data)

    def test_missing_token_has_a_clear_error_and_does_not_call_model(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(llm, "_post_chat") as request:
            result = llm.analyze_image_pair_with_llm("movein.png", "moveout.png", "wall")

        self.assertIsNone(result)
        self.assertIn("api key", llm.get_last_error().lower())
        request.assert_not_called()

    def test_nonexistent_image_is_reported_without_loading_image_packages(self):
        valid, message = llm.validate_image_file("path-that-does-not-exist.png")

        self.assertFalse(valid)
        self.assertIn("missing", message.lower())

    def test_fabricated_quotation_in_model_explanation_is_rejected(self):
        problem = llm._explanation_problem(
            'The agreement says "Tenant must pay a new cleaning fee."', "{}", 
            "The agreement says the tenant must clean the unit."
        )

        self.assertIn("quotation", problem)

    def test_benchmark_explanation_is_optional_and_cannot_add_numeric_estimates(self):
        benchmark = {
            "status": "matched",
            "city": "Bengaluru",
            "locality": "HSR Layout",
            "category": "Painting / Repainting",
            "match_level": "city_locality_category",
            "match_description": "Exact city, locality, and category",
            "position": "above",
            "record_count": 1,
            "lower": 8000,
            "upper": 14000,
            "claimed_amount": 16000,
            "sample_data_warning": "Sample data only.",
        }
        response = {"choices": [{"message": {"content": json.dumps({
            "explanation": "The claim exceeds the range in this sample, which is not a verified quote."
        })}}]}
        with patch.object(llm, "_post_chat", return_value=response) as request:
            explanation = llm.explain_repair_benchmark_with_llm(benchmark)

        self.assertIn("not a verified quote", explanation)
        prompt = request.call_args.args[0]["messages"][0]["content"]
        self.assertIn("do not change the calculation", prompt)
        self.assertIn('"lower": 8000', prompt)

    def test_benchmark_explanation_rejects_model_introduced_numbers(self):
        benchmark = {
            "status": "matched", "city": "Bengaluru", "category": "Painting",
            "position": "above",
            "lower": 8000, "upper": 14000, "claimed_amount": 16000,
        }
        invalid = {"choices": [{"message": {"content": json.dumps({
            "explanation": "This appears to be 20% above the typical local repair cost."
        })}}]}
        with patch.object(llm, "_post_chat", side_effect=[invalid, invalid]) as request:
            explanation = llm.explain_repair_benchmark_with_llm(benchmark)

        self.assertIsNone(explanation)
        self.assertEqual(request.call_count, 2)
        self.assertIn("numeric price estimates", llm.get_last_error())

    def test_hosted_request_uses_the_configured_bearer_token(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"choices": []}'
        with patch.object(llm.urllib.request, "urlopen", return_value=response) as urlopen:
            llm._post_chat({"model": "example/model", "messages": []}, timeout=1)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer test-token-value")

    def test_google_provider_uses_one_multimodal_model_and_compatible_request(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"choices": []}'
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "google",
            "GEMINI_API_KEY": "demo-google-key",
        }), patch.object(llm.urllib.request, "urlopen", return_value=response) as urlopen:
            self.assertEqual(llm.text_model(), "gemini-3.1-flash-lite")
            payload = {
                "model": llm.vision_model(),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these photos."},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,first"}},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,second"}},
                    ],
                }],
                "max_tokens": 1200,
                "temperature": 0,
            }
            llm._post_chat(payload, timeout=1)

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        )
        self.assertEqual(request.headers["Authorization"], "Bearer demo-google-key")
        sent_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent_payload["model"], "gemini-3.1-flash-lite")
        self.assertEqual(
            [part["type"] for part in sent_payload["messages"][0]["content"]],
            ["text", "image_url", "image_url"],
        )

    def test_google_quota_error_is_reported_with_actionable_guidance(self):
        error = llm.urllib.error.HTTPError(
            "https://example.invalid",
            429,
            "Too many requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"Project quota exceeded","status":"RESOURCE_EXHAUSTED"}}'),
        )
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "google",
            "GEMINI_API_KEY": "demo-google-key",
        }), patch.object(llm.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(llm.LLMError) as raised:
                llm._post_chat({"model": llm.text_model(), "messages": []}, timeout=1)

        self.assertIn("quota or rate limit", str(raised.exception).lower())
        self.assertIn("Project quota exceeded", str(raised.exception))

    def test_google_does_not_use_the_hugging_face_token_as_its_key(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "google",
            "GEMINI_API_KEY": "",
            "HF_TOKEN": "test-hf-token",
        }, clear=True):
            self.assertFalse(llm.llm_is_configured())

    def test_google_invalid_api_key_error_has_actionable_guidance(self):
        error = llm.urllib.error.HTTPError(
            "https://example.invalid",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"API key not valid. Please pass a valid API key."}}'),
        )
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "google",
            "GEMINI_API_KEY": "bad-demo-key",
        }), patch.object(llm.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(llm.LLMError) as raised:
                llm._post_chat({"model": llm.text_model(), "messages": []}, timeout=1)

        self.assertIn("API key is invalid", str(raised.exception))
        self.assertIn("GEMINI_API_KEY", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
