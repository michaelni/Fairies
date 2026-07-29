"""The self-identification each role's prompt asks for names one model once.

Posted roles (combiner, triagers, issue investigator) open with
``LLM-<label>`` (vendor prefix stripped, uppercased). The PR reviewer is a
combiner-bound draft: it identifies with the bare label the combiner's
"Draft review from <label>" headers already use — asking drafts for the
``LLM-`` form made combined reviews mix "LLM-GLM-5.2" with "GLM-5.2" for
the same model (e.g. PR #23016).
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


class PromptForTests(unittest.TestCase):
    def test_derived_identity_facts(self) -> None:
        for role, subject, persona, combiner, draft in (
            ("review",             "PR",    "reviewer",     False, True),
            ("code_review",        "PR",    "reviewer",     False, True),
            ("design_review",      "PR",    "reviewer",     False, True),
            ("combiner",           "PR",    "reviewer",     True,  False),
            ("triager",            "PR",    "reviewer",     False, False),
            ("issue_investigator", "issue", "investigator", False, False),
            ("issue_combiner",     "issue", "investigator", True,  False),
            ("issue_triager",      "issue", "investigator", False, False),
        ):
            ctx = llm_prompt.PromptFor(role, "gpt-5.4")
            self.assertEqual(
                (subject, persona, combiner, draft),
                (ctx.subject, ctx.persona, ctx.combiner, ctx.draft),
                role,
            )


class LlmPrefixPromptTests(unittest.TestCase):
    def _prompt(self, role: str, model: str) -> str:
        return llm_prompt.generate_llm_prompt(
            role=role, vendor="openai", model=model, features=set(),
            repo_roots=[], container_repo_mounts=[], reviewer_username="fairy",
        )

    def test_posted_roles_identify_with_the_llm_prefix(self) -> None:
        for role, model, label in (
            ("combiner", "gpt-5.4", "LLM-GPT-5.4"),
            ("triager", "gpt-5.4-mini", "LLM-GPT-5.4-MINI"),
            ("issue_investigator", "zai:glm-5.2", "LLM-GLM-5.2"),
            ("issue_combiner", "gpt-5.4", "LLM-GPT-5.4"),
            ("issue_triager", "gpt-5.4-mini", "LLM-GPT-5.4-MINI"),
        ):
            prompt = self._prompt(role, model)
            self.assertIn(f'Include "{label}"', prompt, role)

    def test_draft_reviewer_identifies_with_the_bare_label(self) -> None:
        for role in llm_prompt.REVIEW_PROMPTS:
            prompt = self._prompt(role, "zai:glm-5.2")
            self.assertIn('Include "GLM-5.2"', prompt, role)
            self.assertNotIn('Include "LLM-', prompt, role)

    def test_combiner_user_text_uses_the_same_labels(self) -> None:
        text = llm_prompt.make_combiner_user_text([
            Review("minor_issues_approve", "a", model="openai:gpt-5.4"),
            Review("major_issues", "b", model="zai:glm-5.2"),
        ])
        self.assertIn("Draft review from GPT-5.4", text)
        self.assertIn("Draft review from GLM-5.2", text)


if __name__ == "__main__":
    unittest.main()
