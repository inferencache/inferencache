"""Tests for temporal validity classification."""

from inferencache.ttl import TTLClass, TTLClassifier


classifier = TTLClassifier()


def test_ephemeral_open_prs():
    assert classifier.classify("What PRs are open right now?") == TTLClass.EPHEMERAL


def test_ephemeral_build_passing():
    assert classifier.classify("Is the build passing?") == TTLClass.EPHEMERAL


def test_ephemeral_build_status():
    assert classifier.classify("What's the current build status?") == TTLClass.EPHEMERAL


def test_session_summarize_file():
    assert classifier.classify("Summarize this file") == TTLClass.SESSION


def test_session_what_does_function_do():
    assert classifier.classify("What does this function do?") == TTLClass.SESSION


def test_session_explain_implementation():
    assert (
        classifier.classify("Explain the implementation in this module")
        == TTLClass.SESSION
    )


def test_session_failing_tests():
    assert (
        classifier.classify("What tests are failing in this repo?")
        == TTLClass.SESSION
    )


def test_time_windowed_langchain_version():
    assert (
        classifier.classify("What's the latest version of langchain?")
        == TTLClass.TIME_WINDOWED
    )


def test_time_windowed_best_practices():
    assert (
        classifier.classify("What are current best practices for RAG?")
        == TTLClass.TIME_WINDOWED
    )


def test_time_windowed_benchmarks():
    assert (
        classifier.classify("Latest benchmark results for GPT-4o")
        == TTLClass.TIME_WINDOWED
    )


def test_permanent_cosine_similarity():
    assert classifier.classify("What is cosine similarity?") == TTLClass.PERMANENT


def test_permanent_regex_email():
    assert (
        classifier.classify("Regex for validating an email address")
        == TTLClass.PERMANENT
    )


def test_permanent_backpropagation():
    assert (
        classifier.classify("How does backpropagation work?")
        == TTLClass.PERMANENT
    )


def test_permanent_git_undo():
    assert (
        classifier.classify("Git command to undo last commit")
        == TTLClass.PERMANENT
    )


def test_empty_prompt_defaults_to_permanent():
    assert classifier.classify("") == TTLClass.PERMANENT


def test_unmatched_prompt_defaults_to_permanent():
    assert classifier.classify("hello world") == TTLClass.PERMANENT
