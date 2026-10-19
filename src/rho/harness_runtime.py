from __future__ import annotations

import json
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    model: str | None
    reasoning_effort: str
    economy_after_tools: int
    strict_after_tools: int
    finalize_after_tools: int
    strict_reminder_every: int


@dataclass(frozen=True)
class ProfileSelection:
    profile: BudgetProfile
    rule_name: str
    guidance: str
    source: str = "rule"
    confidence: float | None = None
    rationale: str = ""
    classifier_model: str | None = None
    classifier_trajectory_id: str | None = None
    classified_profile: str | None = None
    execution_adjustment: str | None = None
    adjuster_action: str | None = None
    adjuster_confidence: float | None = None
    adjuster_rationale: str = ""
    adjuster_model: str | None = None
    adjuster_trajectory_id: str | None = None


@dataclass(frozen=True)
class ConfidenceOverride:
    name: str
    source_profile: str
    below: float
    profile: BudgetProfile


@dataclass(frozen=True)
class ModelClassifierConfig:
    model: str
    reasoning_effort: str
    transport: str
    fallback_profile: str
    rubric_file: Path
    rubric: str
    timeout_s: float
    max_attempts: int
    max_task_chars: int
    max_output_tokens: int
    samples: int
    cross_effort_disagreement_profile: str | None


@dataclass(frozen=True)
class ProfileAdjusterConfig:
    model: str
    reasoning_effort: str
    transport: str
    rubric_file: Path
    rubric: str
    timeout_s: float
    max_attempts: int
    max_task_chars: int
    max_output_tokens: int
    samples: int


@dataclass(frozen=True)
class FreshReviewConfig:
    min_tools: int
    max_tools: int
    timeout_s: float
    profiles: tuple[str, ...]
    reasoning_effort: str | None
    review_tool_budget: int = 8
    continuation_tool_budget: int = 0
    profile_tool_ranges: tuple[tuple[str, int, int], ...] = ()
    review_timed_out: bool = False
    reset_tool_budget: bool = False
    mode: str = "in_place"
    judge_reasoning_effort: str | None = None
    confidence_below: float | None = None
    low_confidence_below: float | None = None
    low_confidence_profile_tool_ranges: tuple[tuple[str, int, int], ...] = ()
    always_profile_tool_ranges: tuple[tuple[str, int, int], ...] = ()
    instructions_file: Path | None = None
    instructions: str | None = None
    any_patterns: tuple[str, ...] = ()

    continuation_summary_chars: int = 0


@dataclass(frozen=True)
class ClassificationRule:
    name: str
    profile: str
    any_patterns: tuple[str, ...]
    all_patterns: tuple[str, ...]
    guidance: str

    def matches(self, prompt: str) -> bool:
        any_ok = not self.any_patterns or any(
            re.search(pattern, prompt, flags=re.IGNORECASE | re.MULTILINE)
            for pattern in self.any_patterns
        )
        all_ok = all(
            re.search(pattern, prompt, flags=re.IGNORECASE | re.MULTILINE)
            for pattern in self.all_patterns
        )
        return any_ok and all_ok


@dataclass(frozen=True)
class HarnessRuntimePolicy:
    source: Path
    profiles: dict[str, BudgetProfile]
    rules: tuple[ClassificationRule, ...]
    default_profile: str
    session_context: str
    routing_runtime_hooks: bool | None
    routing_tool_output_compaction: bool | None
    routing_stage_guidance: bool | None
    routing_tool_budget_enforcement: bool | None
    tool_output_token_limit: int
    tool_output_max_chars: int
    tool_output_head_chars: int
    tool_output_tail_chars: int
    rtk_enabled: bool
    model_instructions_file: Path | None
    external_profile_model_instructions_files: dict[str, Path]
    model_verbosity: str | None
    disabled_features: tuple[str, ...]
    classifier: ModelClassifierConfig | None
    adjuster: ProfileAdjusterConfig | None
    fresh_review: FreshReviewConfig | None
    confidence_overrides: tuple[ConfidenceOverride, ...]

    @classmethod
    def load(cls, path: Path) -> HarnessRuntimePolicy:
        source = path.resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        if int(raw.get("version", 0)) != 1:
            raise ValueError(f"unsupported harness runtime policy version: {raw.get('version')!r}")

        profiles: dict[str, BudgetProfile] = {}
        for name, value in _require_mapping(raw, "profiles").items():
            if not isinstance(value, dict):
                raise ValueError(f"profiles.{name} must be an object")
            reasoning_effort = str(value.get("reasoning_effort", ""))
            if reasoning_effort not in _REASONING_EFFORTS:
                raise ValueError(
                    f"profiles.{name}.reasoning_effort must be one of "
                    f"{sorted(_REASONING_EFFORTS)}"
                )
            thresholds = [
                _positive_int(value, "economy_after_tools", prefix=f"profiles.{name}"),
                _positive_int(value, "strict_after_tools", prefix=f"profiles.{name}"),
                _positive_int(value, "finalize_after_tools", prefix=f"profiles.{name}"),
            ]
            if thresholds != sorted(thresholds) or len(set(thresholds)) != 3:
                raise ValueError(
                    f"profiles.{name} thresholds must be strictly increasing"
                )
            profiles[name] = BudgetProfile(
                name=name,
                model=_optional_model(value.get("model"), f"profiles.{name}.model"),
                reasoning_effort=reasoning_effort,
                economy_after_tools=thresholds[0],
                strict_after_tools=thresholds[1],
                finalize_after_tools=thresholds[2],
                strict_reminder_every=_positive_int(
                    value,
                    "strict_reminder_every",
                    prefix=f"profiles.{name}",
                ),
            )

        default_profile = str(raw.get("default_profile", ""))
        if default_profile not in profiles:
            raise ValueError("default_profile must name a configured profile")

        rules: list[ClassificationRule] = []
        for index, value in enumerate(raw.get("rules", [])):
            if not isinstance(value, dict):
                raise ValueError(f"rules[{index}] must be an object")
            profile = str(value.get("profile", ""))
            if profile not in profiles:
                raise ValueError(f"rules[{index}].profile is not configured")
            rules.append(
                ClassificationRule(
                    name=str(value.get("name") or f"rule_{index}"),
                    profile=profile,
                    any_patterns=_string_tuple(value.get("any", []), f"rules[{index}].any"),
                    all_patterns=_string_tuple(value.get("all", []), f"rules[{index}].all"),
                    guidance=str(value.get("guidance", "")).strip(),
                )
            )

        tool_output = _require_mapping(raw, "tool_output")
        max_chars = _positive_int(tool_output, "max_chars", prefix="tool_output")
        head_chars = _positive_int(tool_output, "head_chars", prefix="tool_output")
        tail_chars = _positive_int(tool_output, "tail_chars", prefix="tool_output")
        if head_chars + tail_chars >= max_chars:
            raise ValueError("tool_output head_chars + tail_chars must be smaller than max_chars")

        codex = raw.get("codex", {})
        if not isinstance(codex, dict):
            raise ValueError("codex must be an object")
        model_instructions_file: Path | None = None
        model_instructions_name = str(codex.get("model_instructions_file", "")).strip()
        if model_instructions_name:
            candidate = (source.parent / model_instructions_name).resolve()
            if not candidate.is_relative_to(source.parent):
                raise ValueError("codex.model_instructions_file must stay in runtime/")
            if not candidate.is_file():
                raise ValueError(
                    f"codex.model_instructions_file does not exist: {candidate}"
                )
            model_instructions_file = candidate
        external_profile_model_instructions_files: dict[str, Path] = {}
        profile_instruction_names = raw.get(
            "external_profile_model_instructions_files",
            {},
        )
        if not isinstance(profile_instruction_names, dict):
            raise ValueError(
                "external_profile_model_instructions_files must be an object"
            )
        for profile_name, filename in profile_instruction_names.items():
            if profile_name not in profiles:
                raise ValueError(
                    "external_profile_model_instructions_files keys must name "
                    "configured profiles"
                )
            candidate = (source.parent / str(filename)).resolve()
            if not candidate.is_relative_to(source.parent) or not candidate.is_file():
                raise ValueError(
                    "external profile model instructions must exist inside runtime/"
                )
            external_profile_model_instructions_files[profile_name] = candidate
        model_verbosity_raw = str(codex.get("model_verbosity", "")).strip()
        if model_verbosity_raw and model_verbosity_raw not in {"low", "medium", "high"}:
            raise ValueError("codex.model_verbosity must be low, medium, or high")

        classifier: ModelClassifierConfig | None = None
        classifier_raw = raw.get("classifier")
        if classifier_raw is not None:
            if not isinstance(classifier_raw, dict):
                raise ValueError("classifier must be an object")
            if str(classifier_raw.get("type", "")) != "model":
                raise ValueError("classifier.type must be 'model'")
            classifier_effort = str(classifier_raw.get("reasoning_effort", "low"))
            if classifier_effort not in _REASONING_EFFORTS:
                raise ValueError("classifier.reasoning_effort is not supported")
            classifier_transport = str(
                classifier_raw.get("transport", "codex")
            ).strip()
            if classifier_transport not in {"anthropic", "codex", "responses"}:
                raise ValueError(
                    "classifier.transport must be anthropic, codex, or responses"
                )
            fallback_profile = str(
                classifier_raw.get("fallback_profile") or default_profile
            )
            if fallback_profile not in profiles:
                raise ValueError("classifier.fallback_profile is not configured")
            rubric_name = str(classifier_raw.get("rubric_file", "")).strip()
            if not rubric_name:
                raise ValueError("classifier.rubric_file is required")
            rubric_file = (source.parent / rubric_name).resolve()
            if not rubric_file.is_relative_to(source.parent) or not rubric_file.is_file():
                raise ValueError("classifier.rubric_file must exist inside runtime/")
            classifier = ModelClassifierConfig(
                model=str(classifier_raw.get("model", "")).strip(),
                reasoning_effort=classifier_effort,
                transport=classifier_transport,
                fallback_profile=fallback_profile,
                rubric_file=rubric_file,
                rubric=rubric_file.read_text(encoding="utf-8").strip(),
                timeout_s=float(classifier_raw.get("timeout_s", 120)),
                max_attempts=_positive_int(
                    classifier_raw,
                    "max_attempts",
                    prefix="classifier",
                ),
                max_task_chars=_positive_int(
                    classifier_raw,
                    "max_task_chars",
                    prefix="classifier",
                ),
                max_output_tokens=_positive_int(
                    classifier_raw,
                    "max_output_tokens",
                    prefix="classifier",
                    default=1024,
                ),
                samples=_positive_int(
                    classifier_raw,
                    "samples",
                    prefix="classifier",
                    default=1,
                ),
                cross_effort_disagreement_profile=(
                    str(
                        classifier_raw.get(
                            "cross_effort_disagreement_profile",
                            "",
                        )
                    ).strip()
                    or None
                ),
            )
            if not classifier.model:
                raise ValueError("classifier.model is required")
            if classifier.timeout_s <= 0:
                raise ValueError("classifier.timeout_s must be positive")
            if (
                classifier.cross_effort_disagreement_profile is not None
                and classifier.cross_effort_disagreement_profile not in profiles
            ):
                raise ValueError(
                    "classifier.cross_effort_disagreement_profile is not configured"
                )

        adjuster: ProfileAdjusterConfig | None = None
        adjuster_raw = raw.get("adjuster")
        if adjuster_raw is not None:
            if not isinstance(adjuster_raw, dict):
                raise ValueError("adjuster must be an object")
            adjuster_effort = str(adjuster_raw.get("reasoning_effort", "medium"))
            if adjuster_effort not in _REASONING_EFFORTS:
                raise ValueError("adjuster.reasoning_effort is not supported")
            adjuster_transport = str(adjuster_raw.get("transport", "responses")).strip()
            if adjuster_transport not in {"codex", "responses"}:
                raise ValueError("adjuster.transport must be codex or responses")
            adjuster_rubric_name = str(adjuster_raw.get("rubric_file", "")).strip()
            if not adjuster_rubric_name:
                raise ValueError("adjuster.rubric_file is required")
            adjuster_rubric_file = (source.parent / adjuster_rubric_name).resolve()
            if (
                not adjuster_rubric_file.is_relative_to(source.parent)
                or not adjuster_rubric_file.is_file()
            ):
                raise ValueError("adjuster.rubric_file must exist inside runtime/")
            adjuster = ProfileAdjusterConfig(
                model=str(adjuster_raw.get("model", "")).strip(),
                reasoning_effort=adjuster_effort,
                transport=adjuster_transport,
                rubric_file=adjuster_rubric_file,
                rubric=adjuster_rubric_file.read_text(encoding="utf-8").strip(),
                timeout_s=float(adjuster_raw.get("timeout_s", 120)),
                max_attempts=_positive_int(
                    adjuster_raw,
                    "max_attempts",
                    prefix="adjuster",
                ),
                max_task_chars=_positive_int(
                    adjuster_raw,
                    "max_task_chars",
                    prefix="adjuster",
                ),
                max_output_tokens=_positive_int(
                    adjuster_raw,
                    "max_output_tokens",
                    prefix="adjuster",
                    default=512,
                ),
                samples=_positive_int(
                    adjuster_raw,
                    "samples",
                    prefix="adjuster",
                    default=1,
                ),
            )
            if not adjuster.model:
                raise ValueError("adjuster.model is required")
            if adjuster.timeout_s <= 0:
                raise ValueError("adjuster.timeout_s must be positive")
            for required_profile in ("compact", "lean", "standard"):
                if required_profile not in profiles:
                    raise ValueError(
                        f"adjuster requires configured profile {required_profile!r}"
                    )

        confidence_overrides: list[ConfidenceOverride] = []
        for index, value in enumerate(raw.get("confidence_overrides", [])):
            if not isinstance(value, dict):
                raise ValueError(f"confidence_overrides[{index}] must be an object")
            source_profile = str(value.get("source_profile", ""))
            if source_profile not in profiles:
                raise ValueError(
                    f"confidence_overrides[{index}].source_profile is not configured"
                )
            below = float(value.get("below", -1))
            if not 0 < below <= 1:
                raise ValueError(
                    f"confidence_overrides[{index}].below must be in (0, 1]"
                )
            reasoning_effort = str(value.get("reasoning_effort", ""))
            if reasoning_effort not in _REASONING_EFFORTS:
                raise ValueError(
                    f"confidence_overrides[{index}].reasoning_effort is unsupported"
                )
            thresholds = [
                _positive_int(
                    value,
                    "economy_after_tools",
                    prefix=f"confidence_overrides[{index}]",
                ),
                _positive_int(
                    value,
                    "strict_after_tools",
                    prefix=f"confidence_overrides[{index}]",
                ),
                _positive_int(
                    value,
                    "finalize_after_tools",
                    prefix=f"confidence_overrides[{index}]",
                ),
            ]
            if thresholds != sorted(thresholds) or len(set(thresholds)) != 3:
                raise ValueError(
                    f"confidence_overrides[{index}] thresholds must be strictly increasing"
                )
            name = str(value.get("name") or f"confidence_override_{index}")
            confidence_overrides.append(
                ConfidenceOverride(
                    name=name,
                    source_profile=source_profile,
                    below=below,
                    profile=BudgetProfile(
                        name=name,
                        model=_optional_model(
                            value.get("model"),
                            f"confidence_overrides[{index}].model",
                        ),
                        reasoning_effort=reasoning_effort,
                        economy_after_tools=thresholds[0],
                        strict_after_tools=thresholds[1],
                        finalize_after_tools=thresholds[2],
                        strict_reminder_every=_positive_int(
                            value,
                            "strict_reminder_every",
                            prefix=f"confidence_overrides[{index}]",
                        ),
                    ),
                )
            )

        fresh_review: FreshReviewConfig | None = None
        fresh_review_raw = raw.get("fresh_review")
        if fresh_review_raw is not None:
            if not isinstance(fresh_review_raw, dict):
                raise ValueError("fresh_review must be an object")
            ranges_raw = fresh_review_raw.get("profile_tool_ranges")
            profile_tool_ranges: tuple[tuple[str, int, int], ...] = ()
            if ranges_raw is not None:
                if not isinstance(ranges_raw, dict) or not ranges_raw:
                    raise ValueError(
                        "fresh_review.profile_tool_ranges must be a non-empty object"
                    )
                parsed_ranges: list[tuple[str, int, int]] = []
                for profile_name, bounds in ranges_raw.items():
                    if profile_name not in profiles:
                        raise ValueError(
                            "fresh_review.profile_tool_ranges references unknown profile: "
                            f"{profile_name}"
                        )
                    if not isinstance(bounds, dict):
                        raise ValueError(
                            "fresh_review.profile_tool_ranges values must be objects"
                        )
                    range_min = _positive_int(
                        bounds,
                        "min_tools",
                        prefix=f"fresh_review.profile_tool_ranges.{profile_name}",
                        default=1,
                    )
                    range_max = _positive_int(
                        bounds,
                        "max_tools",
                        prefix=f"fresh_review.profile_tool_ranges.{profile_name}",
                    )
                    if range_max < range_min:
                        raise ValueError(
                            "fresh_review.profile_tool_ranges max_tools must be >= "
                            f"min_tools for {profile_name}"
                        )
                    parsed_ranges.append((profile_name, range_min, range_max))
                profile_tool_ranges = tuple(parsed_ranges)
                min_tools = min(item[1] for item in profile_tool_ranges)
                max_tools = max(item[2] for item in profile_tool_ranges)
            else:
                min_tools = _positive_int(
                    fresh_review_raw,
                    "min_tools",
                    prefix="fresh_review",
                    default=1,
                )
                max_tools = _positive_int(
                    fresh_review_raw,
                    "max_tools",
                    prefix="fresh_review",
                )
            timeout_s = float(fresh_review_raw.get("timeout_s", 180))
            review_tool_budget = _positive_int(
                fresh_review_raw,
                "review_tool_budget",
                prefix="fresh_review",
                default=8,
            )
            continuation_tool_budget_raw = fresh_review_raw.get(
                "continuation_tool_budget", 0
            )
            if (
                not isinstance(continuation_tool_budget_raw, int)
                or isinstance(continuation_tool_budget_raw, bool)
                or continuation_tool_budget_raw < 0
            ):
                raise ValueError(
                    "fresh_review.continuation_tool_budget must be a non-negative integer"
                )
            profiles_raw = fresh_review_raw.get("profiles")
            continuation_summary_chars = fresh_review_raw.get("continuation_summary_chars", 0)
            if type(continuation_summary_chars) is not int or not 0 <= continuation_summary_chars <= 12000:
                raise ValueError("fresh_review.continuation_summary_chars must be an integer in [0, 12000]")
            if profile_tool_ranges:
                if profiles_raw is not None:
                    raise ValueError(
                        "fresh_review.profiles cannot be combined with "
                        "profile_tool_ranges"
                    )
                fresh_review_profiles = tuple(item[0] for item in profile_tool_ranges)
            elif profiles_raw is None:
                fresh_review_profiles = tuple(
                    name for name in profiles if name != "compact"
                )
            else:
                fresh_review_profiles = _string_tuple(
                    profiles_raw,
                    "fresh_review.profiles",
                )
                if not fresh_review_profiles:
                    raise ValueError("fresh_review.profiles must not be empty")
                unknown_profiles = set(fresh_review_profiles) - set(profiles)
                if unknown_profiles:
                    raise ValueError(
                        "fresh_review.profiles references unknown profiles: "
                        f"{sorted(unknown_profiles)}"
                    )
            review_reasoning_effort_raw = fresh_review_raw.get("reasoning_effort")
            review_reasoning_effort = (
                str(review_reasoning_effort_raw).strip()
                if review_reasoning_effort_raw is not None
                else None
            )
            if (
                review_reasoning_effort is not None
                and review_reasoning_effort not in _REASONING_EFFORTS
            ):
                raise ValueError(
                    "fresh_review.reasoning_effort must be one of "
                    f"{sorted(_REASONING_EFFORTS)}"
                )
            review_timed_out = fresh_review_raw.get("review_timed_out", False)
            if not isinstance(review_timed_out, bool):
                raise ValueError("fresh_review.review_timed_out must be a boolean")
            reset_tool_budget = fresh_review_raw.get("reset_tool_budget", False)
            if not isinstance(reset_tool_budget, bool):
                raise ValueError("fresh_review.reset_tool_budget must be a boolean")
            confidence_below_raw = fresh_review_raw.get("confidence_below")
            confidence_below = (
                float(confidence_below_raw)
                if confidence_below_raw is not None
                else None
            )
            if confidence_below is not None and not 0 < confidence_below <= 1:
                raise ValueError("fresh_review.confidence_below must be in (0, 1]")
            low_confidence_below_raw = fresh_review_raw.get("low_confidence_below")
            low_confidence_below = (
                float(low_confidence_below_raw)
                if low_confidence_below_raw is not None
                else None
            )
            if (
                low_confidence_below is not None
                and not 0 < low_confidence_below <= 1
            ):
                raise ValueError(
                    "fresh_review.low_confidence_below must be in (0, 1]"
                )
            low_confidence_ranges_raw = fresh_review_raw.get(
                "low_confidence_profile_tool_ranges"
            )
            low_confidence_profile_tool_ranges: tuple[
                tuple[str, int, int], ...
            ] = ()
            if low_confidence_ranges_raw is not None:
                if (
                    not isinstance(low_confidence_ranges_raw, dict)
                    or not low_confidence_ranges_raw
                ):
                    raise ValueError(
                        "fresh_review.low_confidence_profile_tool_ranges must "
                        "be a non-empty object"
                    )
                if low_confidence_below is None:
                    raise ValueError(
                        "fresh_review.low_confidence_profile_tool_ranges requires "
                        "low_confidence_below"
                    )
                parsed_low_confidence_ranges: list[tuple[str, int, int]] = []
                for profile_name, bounds in low_confidence_ranges_raw.items():
                    if profile_name not in profiles:
                        raise ValueError(
                            "fresh_review.low_confidence_profile_tool_ranges "
                            f"references unknown profile: {profile_name}"
                        )
                    if not isinstance(bounds, dict):
                        raise ValueError(
                            "fresh_review.low_confidence_profile_tool_ranges "
                            "values must be objects"
                        )
                    range_min = _positive_int(
                        bounds,
                        "min_tools",
                        prefix=(
                            "fresh_review.low_confidence_profile_tool_ranges."
                            f"{profile_name}"
                        ),
                        default=1,
                    )
                    range_max = _positive_int(
                        bounds,
                        "max_tools",
                        prefix=(
                            "fresh_review.low_confidence_profile_tool_ranges."
                            f"{profile_name}"
                        ),
                    )
                    if range_max < range_min:
                        raise ValueError(
                            "fresh_review.low_confidence_profile_tool_ranges "
                            f"max_tools must be >= min_tools for {profile_name}"
                        )
                    parsed_low_confidence_ranges.append(
                        (profile_name, range_min, range_max)
                    )
                low_confidence_profile_tool_ranges = tuple(
                    parsed_low_confidence_ranges
                )
            always_ranges_raw = fresh_review_raw.get(
                "always_profile_tool_ranges"
            )
            always_profile_tool_ranges: tuple[
                tuple[str, int, int], ...
            ] = ()
            if always_ranges_raw is not None:
                if not isinstance(always_ranges_raw, dict) or not always_ranges_raw:
                    raise ValueError(
                        "fresh_review.always_profile_tool_ranges must be a "
                        "non-empty object"
                    )
                parsed_always_ranges: list[tuple[str, int, int]] = []
                for profile_name, bounds in always_ranges_raw.items():
                    if profile_name not in profiles:
                        raise ValueError(
                            "fresh_review.always_profile_tool_ranges references "
                            f"unknown profile: {profile_name}"
                        )
                    if not isinstance(bounds, dict):
                        raise ValueError(
                            "fresh_review.always_profile_tool_ranges values must "
                            "be objects"
                        )
                    range_min = _positive_int(
                        bounds,
                        "min_tools",
                        prefix=(
                            "fresh_review.always_profile_tool_ranges."
                            f"{profile_name}"
                        ),
                        default=1,
                    )
                    range_max = _positive_int(
                        bounds,
                        "max_tools",
                        prefix=(
                            "fresh_review.always_profile_tool_ranges."
                            f"{profile_name}"
                        ),
                    )
                    if range_max < range_min:
                        raise ValueError(
                            "fresh_review.always_profile_tool_ranges max_tools "
                            f"must be >= min_tools for {profile_name}"
                        )
                    parsed_always_ranges.append(
                        (profile_name, range_min, range_max)
                    )
                always_profile_tool_ranges = tuple(parsed_always_ranges)
            review_any_patterns = _string_tuple(
                fresh_review_raw.get("any_patterns", []),
                "fresh_review.any_patterns",
            )
            review_instructions_file: Path | None = None
            review_instructions: str | None = None
            review_instructions_name = str(
                fresh_review_raw.get("instructions_file", "")
            ).strip()
            if review_instructions_name:
                candidate = (source.parent / review_instructions_name).resolve()
                if not candidate.is_relative_to(source.parent):
                    raise ValueError(
                        "fresh_review.instructions_file must stay in runtime/"
                    )
                if not candidate.is_file():
                    raise ValueError(
                        "fresh_review.instructions_file does not exist: "
                        f"{candidate}"
                    )
                review_instructions_file = candidate
                review_instructions = candidate.read_text(encoding="utf-8").strip()
                if not review_instructions:
                    raise ValueError(
                        "fresh_review.instructions_file must not be empty"
                    )
            mode = str(fresh_review_raw.get("mode", "in_place")).strip()
            if mode not in {"in_place", "independent_select"}:
                raise ValueError(
                    "fresh_review.mode must be in_place or independent_select"
                )
            judge_reasoning_effort_raw = fresh_review_raw.get(
                "judge_reasoning_effort"
            )
            judge_reasoning_effort = (
                str(judge_reasoning_effort_raw).strip()
                if judge_reasoning_effort_raw is not None
                else review_reasoning_effort
            )
            if (
                judge_reasoning_effort is not None
                and judge_reasoning_effort not in _REASONING_EFFORTS
            ):
                raise ValueError(
                    "fresh_review.judge_reasoning_effort must be one of "
                    f"{sorted(_REASONING_EFFORTS)}"
                )
            if max_tools < min_tools:
                raise ValueError("fresh_review.max_tools must be >= min_tools")
            if timeout_s <= 0:
                raise ValueError("fresh_review.timeout_s must be positive")
            fresh_review = FreshReviewConfig(
                min_tools=min_tools,
                max_tools=max_tools,
                timeout_s=timeout_s,
                profiles=fresh_review_profiles,
                reasoning_effort=review_reasoning_effort,
                review_tool_budget=review_tool_budget,
                continuation_tool_budget=continuation_tool_budget_raw,
                continuation_summary_chars=continuation_summary_chars,
                profile_tool_ranges=profile_tool_ranges,
                review_timed_out=review_timed_out,
                reset_tool_budget=reset_tool_budget,
                mode=mode,
                judge_reasoning_effort=judge_reasoning_effort,
                confidence_below=confidence_below,
                low_confidence_below=low_confidence_below,
                low_confidence_profile_tool_ranges=(
                    low_confidence_profile_tool_ranges
                ),
                always_profile_tool_ranges=always_profile_tool_ranges,
                instructions_file=review_instructions_file,
                instructions=review_instructions,
                any_patterns=review_any_patterns,
            )

        routing_runtime_hooks_raw = raw.get("routing_runtime_hooks")
        if routing_runtime_hooks_raw is not None and not isinstance(
            routing_runtime_hooks_raw, bool
        ):
            raise ValueError("routing_runtime_hooks must be a boolean when present")
        routing_feature_values: dict[str, bool | None] = {}
        for key in (
            "routing_tool_output_compaction",
            "routing_stage_guidance",
            "routing_tool_budget_enforcement",
        ):
            value = raw.get(key)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean when present")
            routing_feature_values[key] = value

        return cls(
            source=source,
            profiles=profiles,
            rules=tuple(rules),
            default_profile=default_profile,
            session_context=str(raw.get("session_context", "")).strip(),
            routing_runtime_hooks=routing_runtime_hooks_raw,
            routing_tool_output_compaction=routing_feature_values[
                "routing_tool_output_compaction"
            ],
            routing_stage_guidance=routing_feature_values["routing_stage_guidance"],
            routing_tool_budget_enforcement=routing_feature_values[
                "routing_tool_budget_enforcement"
            ],
            tool_output_token_limit=_positive_int(
                tool_output,
                "history_token_limit",
                prefix="tool_output",
            ),
            tool_output_max_chars=max_chars,
            tool_output_head_chars=head_chars,
            tool_output_tail_chars=tail_chars,
            rtk_enabled=bool(tool_output.get("rtk_enabled", False)),
            model_instructions_file=model_instructions_file,
            external_profile_model_instructions_files=(
                external_profile_model_instructions_files
            ),
            model_verbosity=model_verbosity_raw or None,
            disabled_features=_string_tuple(
                codex.get("disabled_features", []),
                "codex.disabled_features",
            ),
            classifier=classifier,
            adjuster=adjuster,
            fresh_review=fresh_review,
            confidence_overrides=tuple(confidence_overrides),
        )

    def select(self, prompt: str) -> ProfileSelection:
        for rule in self.rules:
            if rule.matches(prompt):
                return ProfileSelection(
                    self.profiles[rule.profile],
                    rule.name,
                    rule.guidance,
                )
        return ProfileSelection(self.profiles[self.default_profile], "default", "")

    def select_model_profile(
        self,
        profile_name: str,
        *,
        confidence: float | None,
        rationale: str,
        classifier_model: str,
        classifier_trajectory_id: str,
        vote_counts: dict[str, int] | None = None,
    ) -> ProfileSelection:
        if profile_name not in self.profiles:
            raise ValueError(f"classifier returned unknown profile: {profile_name!r}")
        effective_profile = self.profiles[profile_name]
        execution_adjustment: str | None = None
        disagreement_profile = (
            self.classifier.cross_effort_disagreement_profile
            if self.classifier is not None
            else None
        )
        if disagreement_profile and vote_counts:
            voted_efforts = {
                self.profiles[name].reasoning_effort
                for name, count in vote_counts.items()
                if count > 0 and name in self.profiles
            }
            has_low = bool(voted_efforts & {"minimal", "low"})
            has_high = bool(voted_efforts & {"high", "xhigh"})
            if has_low and has_high and disagreement_profile != profile_name:
                effective_profile = self.profiles[disagreement_profile]
                execution_adjustment = "cross_effort_disagreement"
        if confidence is not None:
            for override in self.confidence_overrides:
                if (
                    override.source_profile == profile_name
                    and confidence < override.below
                ):
                    effective_profile = override.profile
                    execution_adjustment = override.name
                    break
        return ProfileSelection(
            profile=effective_profile,
            rule_name="model_rubric",
            guidance="",
            source="model",
            confidence=confidence,
            rationale=rationale.strip(),
            classifier_model=classifier_model,
            classifier_trajectory_id=classifier_trajectory_id,
            classified_profile=profile_name,
            execution_adjustment=execution_adjustment,
        )

    def fallback_selection(self, reason: str) -> ProfileSelection:
        profile_name = (
            self.classifier.fallback_profile
            if self.classifier is not None
            else self.default_profile
        )
        return ProfileSelection(
            profile=self.profiles[profile_name],
            rule_name="classifier_fallback",
            guidance="",
            source="fallback",
            rationale=reason.strip(),
            classifier_model=self.classifier.model if self.classifier else None,
        )

    def adjust_model_profile(
        self,
        selection: ProfileSelection,
        *,
        action: str,
        confidence: float | None,
        rationale: str,
        adjuster_model: str,
        adjuster_trajectory_id: str,
    ) -> ProfileSelection:
        if action not in {"keep", "compact", "standard"}:
            raise ValueError(f"adjuster returned unknown action: {action!r}")
        effective_profile = selection.profile
        execution_adjustment = selection.execution_adjustment
        if action == "compact" and selection.profile.name != "compact":
            effective_profile = self.profiles["compact"]
            execution_adjustment = "model_adjuster_compact"
        elif action == "standard" and selection.profile.name == "lean":
            effective_profile = self.profiles["standard"]
            execution_adjustment = "model_adjuster_standard"
        else:
            action = "keep"
        return ProfileSelection(
            profile=effective_profile,
            rule_name=selection.rule_name,
            guidance=selection.guidance,
            source=selection.source,
            confidence=selection.confidence,
            rationale=selection.rationale,
            classifier_model=selection.classifier_model,
            classifier_trajectory_id=selection.classifier_trajectory_id,
            classified_profile=selection.classified_profile,
            execution_adjustment=execution_adjustment,
            adjuster_action=action,
            adjuster_confidence=confidence,
            adjuster_rationale=rationale.strip(),
            adjuster_model=adjuster_model,
            adjuster_trajectory_id=adjuster_trajectory_id,
        )

    def classifier_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": sorted(self.profiles),
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "rationale": {
                    "type": "string",
                    "maxLength": 300,
                },
            },
            "required": ["profile", "confidence", "rationale"],
            "additionalProperties": False,
        }

    def classifier_instructions(self, prompt: str) -> str:
        if self.classifier is None:
            raise RuntimeError("runtime policy has no model classifier")
        task = _public_task_description(prompt, self.classifier.max_task_chars)
        profiles = ", ".join(sorted(self.profiles))
        return f"""Classify the task's execution budget using only the general rubric below.

Do not call tools or use remembered historical outcomes. Judge only properties
observable from the supplied task description. Apply the rubric gates in order
and stop at the first match. If required evidence is absent, continue; if no gate
matches, use the default profile.

Allowed profiles: {profiles}

<rubric>
{self.classifier.rubric}
</rubric>

<task-description>
{task}
</task-description>

Return the required JSON object. Keep the rationale under 300 characters.
"""

    def adjuster_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["keep", "compact", "standard"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "rationale": {"type": "string", "maxLength": 300},
            },
            "required": ["action", "confidence", "rationale"],
            "additionalProperties": False,
        }

    def adjuster_instructions(self, prompt: str, baseline_profile: str) -> str:
        if self.adjuster is None:
            raise RuntimeError("runtime policy has no model adjuster")
        task = _public_task_description(prompt, self.adjuster.max_task_chars)
        return f"""Review one baseline execution profile using only the narrow rubric below.

Do not call tools or use remembered historical outcomes. Judge only properties
observable from the supplied task description. The baseline profile came from a
separate general classifier. Return `keep` unless one adjustment gate clearly
applies. Never change a baseline `deep` or `research` task to `standard`.

Baseline profile: {baseline_profile}

<rubric>
{self.adjuster.rubric}
</rubric>

<task-description>
{task}
</task-description>

Return the required JSON object. Keep the rationale under 300 characters.
"""

    def workspace_selection_payload(
        self,
        selection: ProfileSelection,
        *,
        reasoning_effort_override: str | None = None,
    ) -> dict[str, Any]:
        profile = selection.profile
        model_instructions_file = (
            self.external_profile_model_instructions_files.get(profile.name)
            or self.model_instructions_file
        )
        payload = {
            "profile": profile.name,
            "model": profile.model,
            "classified_profile": selection.classified_profile or profile.name,
            "execution_adjustment": selection.execution_adjustment,
            "classification_rule": selection.rule_name,
            "classification_source": selection.source,
            "classifier_confidence": selection.confidence,
            "classifier_rationale": selection.rationale,
            "classifier_model": selection.classifier_model,
            "classifier_trajectory_id": selection.classifier_trajectory_id,
            "adjuster_action": selection.adjuster_action,
            "adjuster_confidence": selection.adjuster_confidence,
            "adjuster_rationale": selection.adjuster_rationale,
            "adjuster_model": selection.adjuster_model,
            "adjuster_trajectory_id": selection.adjuster_trajectory_id,
            "reasoning_effort": profile.reasoning_effort,
            "economy_after_tools": profile.economy_after_tools,
            "strict_after_tools": profile.strict_after_tools,
            "finalize_after_tools": profile.finalize_after_tools,
            "strict_reminder_every": profile.strict_reminder_every,
            "tool_output_max_chars": self.tool_output_max_chars,
            "tool_output_head_chars": self.tool_output_head_chars,
            "tool_output_tail_chars": self.tool_output_tail_chars,
            "rtk_enabled": self.rtk_enabled,
            "session_context": self.session_context,
            "routing_runtime_hooks": self.routing_runtime_hooks,
            "routing_tool_output_compaction": self.routing_tool_output_compaction,
            "routing_stage_guidance": self.routing_stage_guidance,
            "routing_tool_budget_enforcement": self.routing_tool_budget_enforcement,
            "task_guidance": selection.guidance,
            "model_instructions": (
                model_instructions_file.read_text(encoding="utf-8").strip()
                if model_instructions_file is not None
                else ""
            ),
        }
        if reasoning_effort_override is not None:
            payload["profile_reasoning_effort"] = profile.reasoning_effort
            payload["reasoning_effort"] = reasoning_effort_override
            payload["reasoning_effort_override"] = reasoning_effort_override
        return payload


def load_runtime_policy(harness_root: Path) -> HarnessRuntimePolicy | None:
    path = harness_root / "runtime" / "policy.json"
    return HarnessRuntimePolicy.load(path) if path.is_file() else None


def runtime_state_dir(state_root: Path, workspace: Path) -> Path:
    """Return a stable external state path for one ephemeral solve workspace."""

    key = hashlib.sha256(str(workspace.resolve()).encode("utf-8")).hexdigest()[:24]
    return state_root.resolve() / key


def write_workspace_selection(
    state_root: Path,
    workspace: Path,
    policy: HarnessRuntimePolicy,
    selection: ProfileSelection,
    *,
    reasoning_effort_override: str | None = None,
) -> Path:
    # Policy/profile state is deliberately kept outside the model-visible
    # workspace. The hook can recover it from cwd plus RHO_RUNTIME_STATE_ROOT.
    state_dir = runtime_state_dir(state_root, workspace)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "selection.json"
    path.write_text(
        json.dumps(
            policy.workspace_selection_payload(
                selection,
                reasoning_effort_override=reasoning_effort_override,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def render_runtime_codex_config(
    base_config: Path,
    *,
    output: Path,
    policy: HarnessRuntimePolicy,
    hook_script: Path,
    rtk_binary: Path | None,
    runtime_state_root: Path,
) -> Path:
    """Create the per-run Codex layer that enforces a harness runtime policy."""

    hook_command = [
        "RTK_TELEMETRY_DISABLED=1",
        f"RHO_RTK_BIN={_shell_quote(str(rtk_binary.resolve()))}"
        if rtk_binary is not None
        else "RHO_RTK_BIN=",
        f"RHO_RUNTIME_STATE_ROOT={_shell_quote(str(runtime_state_root.resolve()))}",
        "/usr/bin/python3",
        _shell_quote(str(hook_script.resolve())),
    ]
    handler_command = " ".join(hook_command)
    generated_root = f"""# Generated from {policy.source}; do not edit this run copy.
tool_output_token_limit = {policy.tool_output_token_limit}
"""
    if policy.model_instructions_file is not None:
        generated_root += (
            "model_instructions_file = "
            f"{_toml_string(str(policy.model_instructions_file))}\n"
        )
    if policy.model_verbosity is not None:
        generated_root += f"model_verbosity = {_toml_string(policy.model_verbosity)}\n"
    generated_root += "\n"
    base_text = base_config.read_text(encoding="utf-8").rstrip() + "\n"
    if policy.disabled_features:
        base_text = _merge_disabled_codex_features(
            base_text, policy.disabled_features
        )
    appended = f"""
[[hooks.SessionStart]]
matcher = "^startup$"

[[hooks.SessionStart.hooks]]
type = "command"
command = {_toml_string(handler_command)}
timeout = 10
additionalContextLimit = 1200

[[hooks.PreToolUse]]

[[hooks.PreToolUse.hooks]]
type = "command"
command = {_toml_string(handler_command)}
timeout = 10
additionalContextLimit = 1200

[[hooks.PostToolUse]]

[[hooks.PostToolUse.hooks]]
type = "command"
command = {_toml_string(handler_command)}
timeout = 10
additionalContextLimit = 1800
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = generated_root + base_text + appended
    tomllib.loads(rendered)
    output.write_text(rendered, encoding="utf-8")
    return output


def _merge_disabled_codex_features(
    base_text: str, disabled_features: tuple[str, ...]
) -> str:
    """Merge disabled feature flags without emitting a duplicate TOML table."""

    lines = base_text.splitlines()
    header = re.compile(r"^\s*\[features\]\s*(?:#.*)?$")
    any_table = re.compile(r"^\s*\[+")
    start = next((index for index, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["[features]", *(f"{name} = false" for name in disabled_features)])
        return "\n".join(lines).rstrip() + "\n"

    end = next(
        (index for index in range(start + 1, len(lines)) if any_table.match(lines[index])),
        len(lines),
    )
    disabled = set(disabled_features)
    found: set[str] = set()
    assignment = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=")
    for index in range(start + 1, end):
        match = assignment.match(lines[index])
        if match is None or match.group(1) not in disabled:
            continue
        name = match.group(1)
        lines[index] = f"{name} = false"
        found.add(name)
    lines[end:end] = [
        f"{name} = false" for name in disabled_features if name not in found
    ]
    return "\n".join(lines).rstrip() + "\n"


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_model(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string when set")
    return value.strip()


def _positive_int(
    raw: dict[str, Any],
    key: str,
    *,
    prefix: str,
    default: int | None = None,
) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{prefix}.{key} must be a positive integer")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    for pattern in value:
        re.compile(pattern)
    return tuple(value)


def _public_task_description(prompt: str, max_chars: int) -> str:
    """Remove benchmark scaffolding and identifiers before model classification."""
    task_match = re.search(r"(?im)^## Task\s*$", prompt)
    if task_match:
        start = task_match.end()
        finishing_match = re.search(r"(?im)^## Finishing\s*$", prompt[start:])
        end = start + finishing_match.start() if finishing_match else len(prompt)
        description = prompt[start:end].strip()
    else:
        description = re.sub(
            r"(?im)^#\s*Terminal-Bench[^\n]*\n?",
            "",
            prompt.strip(),
            count=1,
        ).strip()
    if len(description) <= max_chars:
        return description
    head_chars = max_chars * 3 // 4
    tail_chars = max_chars - head_chars
    return (
        description[:head_chars].rstrip()
        + "\n\n[task description truncated in the middle]\n\n"
        + description[-tail_chars:].lstrip()
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
