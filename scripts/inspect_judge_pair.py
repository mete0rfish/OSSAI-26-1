"""Gemma 후보 한 쌍을 정체를 가린 채 학습자에게 출력한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from verifiable_ai_workflow.judge_comparison import (
    load_individual_human_label,
    validate_individual_human_label,
)
from verifiable_ai_workflow.open_cqa_candidates import (
    candidate_set_sha256,
    load_candidate_pairs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--human-label", type=Path)
    parser.add_argument(
        "--reveal",
        action="store_true",
        help="검증된 사람 사전 label 뒤에 기대 답과 후보 출처를 공개",
    )
    args = parser.parse_args()
    if args.reveal and args.human_label is None:
        raise SystemExit("--reveal에는 잠근 --human-label이 필요합니다")
    try:
        pairs = load_candidate_pairs(args.candidates)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"candidate-results.jsonl을 확인하세요: {exc}") from exc
    if not 1 <= args.number <= len(pairs):
        raise SystemExit(f"--number는 1부터 {len(pairs)}까지입니다")
    candidate_set_hash = candidate_set_sha256(pairs)
    pair = pairs[args.number - 1]
    print(f"[후보 세트 SHA-256] {candidate_set_hash}")
    print(f"[평가표 ID] {pair.pair_id}")
    print(f"[차트] {pair.image_path}")
    print(f"[질문] {pair.question}")
    print(f"[후보 A] {pair.candidate_a}")
    print(f"[후보 B] {pair.candidate_b}")
    if args.human_label is None:
        print("[학습자 선택] candidate_a / tie / candidate_b 중 하나를 고릅니다")
    if args.human_label is not None:
        try:
            human_label = load_individual_human_label(args.human_label)
            validate_individual_human_label(
                human_label,
                pairs,
                candidate_set_hash,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if human_label.pair_number != args.number:
            raise SystemExit("사람 사전 label의 pair_number가 선택한 pair와 다릅니다")
        print(f"[사람 사전 label 검증 완료] {human_label.label}")
        if args.reveal:
            print(f"[기대 답] {pair.reference_answer}")
            print(f"[후보 A 출처] {pair.candidate_a_source}")
            print(f"[후보 B 출처] {pair.candidate_b_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
