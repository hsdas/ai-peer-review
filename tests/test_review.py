import pytest
from unittest.mock import patch, MagicMock

from ai_peer_review.review import (
    get_review_prompt,
    get_metareview_prompt,
    process_paper,
    generate_meta_review,
)


class TestReview:
    def test_get_review_prompt(self):
        # Test with a sample paper text
        paper_text = "This is a sample paper about neuroscience."
        prompt = get_review_prompt(paper_text)

        # Check that the prompt contains the paper text
        assert paper_text in prompt

        # Check that the prompt contains the required instructions
        assert "neuroscientist and expert in brain imaging" in prompt
        assert "provide a thorough and critical review" in prompt
        assert "summary of the study" in prompt
        assert "point-by-point analysis" in prompt

    def test_get_metareview_prompt(self):
        # Test with sample reviews
        reviews = ["Review 1 content", "Review 2 content"]
        prompt = get_metareview_prompt(reviews)

        # Check that the prompt contains all reviews with NATO names
        assert "Review from alfa:\n\nReview 1 content" in prompt
        assert "Review from bravo:\n\nReview 2 content" in prompt

        # Check that the prompt contains the required instructions
        assert "meta-review" in prompt
        assert "common points" in prompt
        assert "specific concerns" in prompt
        assert "CONCERNS_TABLE_DATA" in prompt

    @patch('ai_peer_review.review.get_client_factories')
    @patch('ai_peer_review.review.extract_text_from_pdf')
    def test_process_paper(self, mock_extract_text, mock_get_factories):
        # Mock PDF extraction
        mock_extract_text.return_value = "Paper content"

        # Mock the local client factory so no real server is contacted.
        # One shared mock client backs both reviewers.
        mock_client = MagicMock()
        mock_client.generate.return_value = "Local review"
        mock_factory = MagicMock(return_value=mock_client)
        mock_get_factories.return_value = {
            "reviewer-1": mock_factory,
            "reviewer-2": mock_factory,
        }

        # Request a subset of the configured reviewers
        models = ["reviewer-1", "reviewer-2"]
        reviews = process_paper("test.pdf", models)

        # PDF text extraction ran once
        mock_extract_text.assert_called_once_with("test.pdf")

        # The factory table was consulted once
        mock_get_factories.assert_called_once()

        # Each requested reviewer built a client and generated once
        assert mock_factory.call_count == 2
        assert mock_client.generate.call_count == 2

        # Results are keyed by reviewer name
        assert set(reviews.keys()) == {"reviewer-1", "reviewer-2"}
        assert reviews["reviewer-1"] == "Local review"
        assert reviews["reviewer-2"] == "Local review"

    @patch('ai_peer_review.review.get_client_factories')
    @patch('ai_peer_review.review.extract_text_from_pdf')
    def test_process_paper_filters_unknown_models(self, mock_extract_text, mock_get_factories):
        # Reviewer names not in the factory table are silently skipped (the
        # CLI validates against get_available_models first, but process_paper
        # stays defensive).
        mock_extract_text.return_value = "Paper content"
        mock_client = MagicMock()
        mock_client.generate.return_value = "Local review"
        mock_get_factories.return_value = {
            "reviewer-1": MagicMock(return_value=mock_client),
        }

        reviews = process_paper("test.pdf", ["reviewer-1", "does-not-exist"])

        assert set(reviews.keys()) == {"reviewer-1"}
        mock_client.generate.assert_called_once()

    @patch('ai_peer_review.review.get_meta_reviewer_client')
    def test_generate_meta_review(self, mock_get_meta_client):
        # Mock the configured local meta-reviewer client
        mock_client = MagicMock()
        mock_client.generate.return_value = "Meta-review content"
        mock_get_meta_client.return_value = mock_client

        # Test with sample reviews
        reviews = {
            "reviewer-1": "Review 1 content",
            "reviewer-2": "Review 2 content",
        }

        meta_review, mapping = generate_meta_review(reviews)

        # The meta-reviewer client was built once and used once
        mock_get_meta_client.assert_called_once()
        mock_client.generate.assert_called_once()

        # The meta-review text is returned verbatim (after stripping the
        # CONCERNS_TABLE_DATA section, which is absent here)
        assert meta_review == "Meta-review content"

        # NATO-to-reviewer mapping was created for every reviewer
        assert len(mapping) == 2
        assert set(mapping.values()) == {"reviewer-1", "reviewer-2"}
        assert set(mapping.keys()) == {"alfa", "bravo"}
