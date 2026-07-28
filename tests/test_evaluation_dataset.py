import json
from pathlib import Path


DATASET_PATH = Path("evaluation/evaluation_dataset.json")
EXPECTED_ABSTENTION = "I don't know based on the uploaded documents."


def load_dataset() -> dict:
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_evaluation_dataset_has_required_metadata() -> None:
    dataset = load_dataset()

    assert dataset["schema_version"] == "1.0"
    assert dataset["dataset_version"] == "7A"
    assert dataset["description"]
    assert dataset["source_documents"]
    assert dataset["examples"]


def test_evaluation_example_ids_are_unique() -> None:
    examples = load_dataset()["examples"]
    example_ids = [example["id"] for example in examples]

    assert len(example_ids) == len(set(example_ids))


def test_dataset_contains_answerable_and_unanswerable_examples() -> None:
    examples = load_dataset()["examples"]

    assert any(example["answerable"] for example in examples)
    assert any(not example["answerable"] for example in examples)


def test_all_examples_have_required_fields() -> None:
    required_fields = {
        "id",
        "question",
        "answerable",
        "expected_documents",
        "expected_pages",
        "expected_page_labels",
        "required_answer_points",
    }

    for example in load_dataset()["examples"]:
        assert required_fields.issubset(example)
        assert example["id"]
        assert example["question"]
        assert isinstance(example["answerable"], bool)
        assert isinstance(example["expected_documents"], list)
        assert isinstance(example["expected_pages"], list)
        assert isinstance(example["expected_page_labels"], list)
        assert isinstance(example["required_answer_points"], list)
        assert len(example["expected_pages"]) == len(
            example["expected_page_labels"]
        )


def test_answerable_examples_have_expected_evidence() -> None:
    examples = load_dataset()["examples"]

    for example in examples:
        if example["answerable"]:
            assert example["expected_documents"]
            assert example["expected_pages"]
            assert example["expected_page_labels"]
            assert example["required_answer_points"]


def test_unanswerable_examples_have_no_expected_evidence() -> None:
    examples = load_dataset()["examples"]

    for example in examples:
        if not example["answerable"]:
            assert example["expected_documents"] == []
            assert example["expected_pages"] == []
            assert example["expected_page_labels"] == []
            assert example["required_answer_points"] == []
            assert example["expected_response"] == EXPECTED_ABSTENTION