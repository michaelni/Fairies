"""The posted-message prefix names the producing model.

Pins the ``LLM-<MODEL>`` instruction: every role's developer prompt tells
the model to include ``LLM-<its own label>`` (vendor prefix stripped,
uppercased) so posted reviews and replies are attributable, and the
combiner user text labels drafts with the same helper.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402
from llm_review_api import Review  # noqa: E402


class ModelLabelTests(unittest.TestCase):
    def test_vendor_prefix_stripped_and_uppercased(self) -> None:
        self.assertEqual("GLM-5.2", llm_prompt.model_label("zai:glm-5.2"))
        self.assertEqual("GPT-5.4", llm_prompt.model_label("gpt-5.4"))
        self.assertEqual("UNKNOWN", llm_prompt.model_label(""))


class LlmPrefixPromptTests(unittest.TestCase):
    def _prompt(self, role: str, model: str) -> str:
        return llm_prompt.generate_llm_prompt(
            role=role, vendor="openai", model=model, features=set(),
            repo_roots=[], container_repo_mounts=[], reviewer_username="bot",
        )

    def test_each_role_prompt_carries_its_model_prefix(self) -> None:
        for role, model, label in (
            ("reviewer", "zai:glm-5.2", "LLM-GLM-5.2"),
            ("combiner", "gpt-5.4", "LLM-GPT-5.4"),
            ("triager", "gpt-5.4-mini", "LLM-GPT-5.4-MINI"),
        ):
            prompt = self._prompt(role, model)
            self.assertIn(f'Include "{label}"', prompt)
            self.assertNotIn('Include "LLM"', prompt)

    def test_combiner_user_text_uses_the_same_labels(self) -> None:
        text = llm_prompt.make_combiner_user_text([
            Review("minor_issues_approve", "a", model="openai:gpt-5.4"),
            Review("major_issues", "b", model="zai:glm-5.2"),
        ])
        self.assertIn("Draft review from GPT-5.4", text)
        self.assertIn("Draft review from GLM-5.2", text)


if __name__ == "__main__":
    unittest.main()
