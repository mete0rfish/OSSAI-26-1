import pytest

from verifiable_ai_workflow.model_identity import canonicalize_litellm_actual_model


@pytest.mark.parametrize(
    ("reported", "expected_result"),
    [
        ("google/gemma-4-31b-it", "google/gemma-4-31b-it"),
        ("nvidia_nim/google/gemma-4-31b-it", "google/gemma-4-31b-it"),
        ("other/google/gemma-4-31b-it", "other/google/gemma-4-31b-it"),
        (
            "nvidia_nim/nvidia_nim/google/gemma-4-31b-it",
            "nvidia_nim/nvidia_nim/google/gemma-4-31b-it",
        ),
        ("nvidia_nim/google/gemma-4-31b-it-v2", "nvidia_nim/google/gemma-4-31b-it-v2"),
        (None, None),
    ],
)
def test_only_exact_litellm_transport_prefix_is_canonicalized(
    reported: str | None,
    expected_result: str | None,
) -> None:
    assert (
        canonicalize_litellm_actual_model(
            reported,
            requested_model="nvidia_nim/google/gemma-4-31b-it",
            expected_actual_model="google/gemma-4-31b-it",
        )
        == expected_result
    )
