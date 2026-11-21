import pytest
import os
import json
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from ai_peer_review.cli import cli, main


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_version(self):
        # Test that the CLI version option works
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    # --- config subcommands -------------------------------------------------

    @patch('ai_peer_review.cli.set_local_model')
    def test_config_add(self, mock_set_local_model):
        result = self.runner.invoke(cli, [
            "config", "add", "reviewer-1", "--model", "qwen3:32b",
        ])
        assert result.exit_code == 0
        mock_set_local_model.assert_called_once()
        args, kwargs = mock_set_local_model.call_args
        assert args[0] == "reviewer-1"
        assert args[1] == "qwen3:32b"
        assert "reviewer-1" in result.output

    @patch('ai_peer_review.cli.set_local_server')
    def test_config_server(self, mock_set_local_server):
        result = self.runner.invoke(cli, [
            "config", "server", "--base-url", "http://localhost:1234/v1",
        ])
        assert result.exit_code == 0
        mock_set_local_server.assert_called_once()
        _, kwargs = mock_set_local_server.call_args
        assert kwargs["base_url"] == "http://localhost:1234/v1"
        assert "http://localhost:1234/v1" in result.output

    @patch('ai_peer_review.cli.set_meta_reviewer')
    def test_config_meta(self, mock_set_meta):
        result = self.runner.invoke(cli, [
            "config", "meta", "--model", "qwen3:32b",
        ])
        assert result.exit_code == 0
        mock_set_meta.assert_called_once()
        args, _ = mock_set_meta.call_args
        assert args[0] == "qwen3:32b"

    @patch('ai_peer_review.cli.remove_local_model', return_value=True)
    def test_config_remove(self, mock_remove):
        result = self.runner.invoke(cli, ["config", "remove", "reviewer-1"])
        assert result.exit_code == 0
        mock_remove.assert_called_once()
        assert "removed" in result.output

    @patch('ai_peer_review.cli.load_config')
    def test_config_show(self, mock_load_config):
        mock_load_config.return_value = {
            "local_server": {"base_url": "http://localhost:11434/v1", "api_key": "local"},
            "local_models": {"reviewer-1": {"model": "qwen3:32b"}},
            "meta_reviewer": {"model": "qwen3:32b"},
        }
        result = self.runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "local_models" in result.output
        assert "qwen3:32b" in result.output

    # --- list-models --------------------------------------------------------

    @patch('ai_peer_review.cli.get_local_models')
    def test_list_models(self, mock_get_local_models):
        mock_get_local_models.return_value = {
            "reviewer-1": {"model": "qwen3:32b"},
            "reviewer-2": {"model": "deepseek-r1:70b"},
        }
        result = self.runner.invoke(cli, ["list-models"])
        assert result.exit_code == 0
        assert "reviewer-1" in result.output
        assert "reviewer-2" in result.output

    # --- review -------------------------------------------------------------

    @patch('ai_peer_review.cli.get_available_models', return_value=["reviewer-1", "reviewer-2"])
    @patch('ai_peer_review.cli.process_paper')
    def test_review_command_no_meta(self, mock_process_paper, mock_avail):
        mock_process_paper.return_value = {"reviewer-1": "Review content"}

        with self.runner.isolated_filesystem():
            with open("test.pdf", "w") as f:
                f.write("Dummy PDF content")

            result = self.runner.invoke(cli, [
                "review", "test.pdf",
                "--no-meta-review",
                "--models", "reviewer-1",
            ])

            assert result.exit_code == 0
            assert "Processing paper: test.pdf" in result.output
            assert "Selected reviewers: reviewer-1" in result.output
            assert os.path.exists("papers/test")
            assert os.path.exists("papers/test/review_reviewer-1.md")

            # No meta-review generated
            assert "Generating meta-review" not in result.output
            assert not os.path.exists("papers/test/meta_review.md")
            assert not os.path.exists("papers/test/results.json")

    @patch('ai_peer_review.cli.get_meta_reviewer_client')
    @patch('ai_peer_review.cli.get_available_models', return_value=["reviewer-1", "reviewer-2"])
    @patch('ai_peer_review.cli.generate_meta_review')
    @patch('ai_peer_review.cli.process_paper')
    def test_review_command_with_meta(
        self, mock_process_paper, mock_generate_meta, mock_avail, mock_meta_client
    ):
        mock_process_paper.return_value = {"reviewer-1": "Review content"}
        mock_generate_meta.return_value = (
            "Meta-review content",
            {"alfa": "reviewer-1"},
        )
        # The concerns-extraction call returns valid JSON
        mock_meta_instance = MagicMock()
        mock_meta_instance.generate.return_value = (
            '{"concerns": [{"concern": "Small sample", "reviewer-1": true}]}'
        )
        mock_meta_client.return_value = mock_meta_instance

        with self.runner.isolated_filesystem():
            with open("test.pdf", "w") as f:
                f.write("Dummy PDF content")

            result = self.runner.invoke(cli, [
                "review", "test.pdf",
                "--models", "reviewer-1",
            ])

            assert result.exit_code == 0
            assert "Generating meta-review" in result.output
            assert os.path.exists("papers/test/meta_review.md")
            assert os.path.exists("papers/test/results.json")
            assert os.path.exists("papers/test/concerns_table.csv")

            with open("papers/test/results.json", "r") as f:
                results = json.load(f)
                assert "individual_reviews" in results
                assert "meta_review" in results
                assert "reviewer_to_model" in results

    @patch('ai_peer_review.cli.get_available_models', return_value=["reviewer-1"])
    def test_review_command_invalid_models(self, mock_avail):
        with self.runner.isolated_filesystem():
            with open("test.pdf", "w") as f:
                f.write("Dummy PDF content")

            result = self.runner.invoke(cli, [
                "review", "test.pdf",
                "--models", "invalid-model1,invalid-model2",
            ])

            assert result.exit_code == 0
            assert "No valid models specified" in result.output

    @patch('ai_peer_review.cli.cli')
    def test_main(self, mock_cli):
        main()
        mock_cli.assert_called_once()
