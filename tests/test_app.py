"""Exercise explanation requests without network access or model credentials."""

import json
import os
from pathlib import Path
import runpy
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import anthropic
import httpx2


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        streamlit = ModuleType("streamlit")
        streamlit.secrets = {"ANTHROPIC_API_KEY": "offline-test"}
        streamlit.title = Mock()
        streamlit.write = Mock()
        streamlit.number_input = lambda label, **kwargs: kwargs["value"]
        streamlit.text_input = lambda label, **kwargs: kwargs["value"]
        streamlit.selectbox = lambda label, options: options[0]
        streamlit.checkbox = lambda label: False
        streamlit.button = lambda label: False
        simulation = ModuleType("policyengine_us")
        simulation.Simulation = Mock()
        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = ModuleType("matplotlib.pyplot")
        modules = {
            "streamlit": streamlit,
            "policyengine_us": simulation,
            "networkx": ModuleType("networkx"),
            "matplotlib": matplotlib,
            "matplotlib.pyplot": matplotlib.pyplot,
        }
        with (
            patch.dict("sys.modules", modules),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline-test"}),
            patch.object(anthropic, "Anthropic", return_value=self.client),
        ):
            app = runpy.run_path(str(APP_PATH))
        self.get_explanation = app["get_explanation"]

    def explain(self):
        return self.get_explanation("snap", 200, "snap = 200")

    def test_reads_all_text_after_thinking(self):
        self.client.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking="Private reasoning"),
                SimpleNamespace(type="text", text="Your benefit is "),
                SimpleNamespace(type="text", text="$200."),
            ],
        )

        self.assertEqual(self.explain(), "Your benefit is $200.")

    def test_refusal_is_checked_before_reading_content(self):
        class Refusal:
            stop_reason = "refusal"

            @property
            def content(self):
                raise AssertionError("Refused content must not be read")

        self.client.messages.create.return_value = Refusal()

        self.assertEqual(
            self.explain(),
            "Failed to get explanation: Claude declined this request.",
        )

    def test_empty_and_non_text_responses_report_failure(self):
        for content in (
            [],
            [SimpleNamespace(type="thinking", thinking="Private reasoning")],
            [SimpleNamespace(type="text", text="")],
        ):
            with self.subTest(content=content):
                self.client.messages.create.return_value = SimpleNamespace(
                    stop_reason="max_tokens", content=content
                )
                self.assertEqual(
                    self.explain(),
                    "Failed to get explanation: the response contained no text "
                    "(stop reason: max_tokens).",
                )

    def test_api_error_reports_failure(self):
        self.client.messages.create.side_effect = anthropic.APIConnectionError(
            message="Offline connection failure",
            request=httpx2.Request("POST", "https://offline.invalid/v1/messages"),
        )

        self.assertEqual(
            self.explain(), "Failed to get explanation: Offline connection failure"
        )

    def test_sdk_serializes_supported_request_without_network(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx2.Response(
                200,
                json={
                    "id": "msg_offline",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Private reasoning",
                            "signature": "offline-signature",
                        },
                        {"type": "text", "text": "Your benefit is $200."},
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            )

        with anthropic.Anthropic(
            api_key="offline-test",
            base_url="https://offline.invalid",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        ) as client:
            with patch.dict(self.get_explanation.__globals__, {"client": client}):
                self.assertEqual(self.explain(), "Your benefit is $200.")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/messages")
        body = json.loads(requests[0].content)
        self.assertEqual(
            set(body), {"model", "max_tokens", "thinking", "output_config", "messages"}
        )
        self.assertEqual(body["model"], "claude-sonnet-5-5")
        self.assertEqual(body["max_tokens"], 16000)
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "low"})
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        prompt = body["messages"][0]["content"]
        self.assertIn("'snap'", prompt)
        self.assertIn("snap = 200", prompt)
        self.assertNotIn("\n\nHuman:", prompt)
        self.assertNotIn("\n\nAssistant:", prompt)


if __name__ == "__main__":
    unittest.main()
