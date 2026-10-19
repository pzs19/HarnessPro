from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICING_AS_OF = "2026-08-14"


@dataclass(frozen=True)
class TokenPrice:
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    cache_write_usd_per_million: float | None = None
    pricing_source: str = PRICING_SOURCE
    pricing_as_of: str = PRICING_AS_OF
    estimate_note: str = (
        "Published OpenAI standard rates; provider-specific discounts/surcharges "
        "and per-request long-context multipliers are not available in aggregate "
        "usage events."
    )


# Standard synchronous API rates. Azure/provider invoices can differ from the
# public OpenAI API price sheet, so reports deliberately call this an estimate.
_PRICES: dict[str, TokenPrice] = {
    "gpt-5.5": TokenPrice(
        input_usd_per_million=5.0,
        cached_input_usd_per_million=0.5,
        output_usd_per_million=30.0,
    ),
    "gpt-5.4-mini": TokenPrice(
        input_usd_per_million=0.75,
        cached_input_usd_per_million=0.075,
        output_usd_per_million=4.5,
    ),
    "gpt-5.4-nano": TokenPrice(
        input_usd_per_million=0.2,
        cached_input_usd_per_million=0.02,
        output_usd_per_million=1.25,
    ),
    "gpt-5-mini": TokenPrice(
        input_usd_per_million=0.25,
        cached_input_usd_per_million=0.025,
        output_usd_per_million=2.0,
    ),
    "gpt-5.6-luna": TokenPrice(
        input_usd_per_million=0.2,
        cached_input_usd_per_million=0.02,
        cache_write_usd_per_million=0.25,
        output_usd_per_million=1.2,
    ),
    "glm-5.3": TokenPrice(
        input_usd_per_million=1.4,
        cached_input_usd_per_million=0.26,
        output_usd_per_million=4.4,
        pricing_source="https://docs.z.ai/guides/overview/pricing",
        pricing_as_of="2026-09-04",
        estimate_note=(
            "Official GLM-5.3 list price. Cached-input storage is temporarily "
            "free and is not represented because the provider usage response "
            "does not expose stored-token duration."
        ),
    ),
    "glm-5.3-flash": TokenPrice(
        input_usd_per_million=0.15,
        cached_input_usd_per_million=0.03,
        output_usd_per_million=0.5,
        pricing_source="https://docs.z.ai/guides/overview/pricing",
        pricing_as_of="2026-09-04",
        estimate_note=(
            "Official GLM-5.3-Flash list price. The temporary 50% promotion "
            "ending September 9, 2026 is deliberately ignored so experiments "
            "remain comparable after the promotion ends. Cached-input storage "
            "is temporarily free and is not represented."
        ),
    ),
    "deepseek-ai/deepseek-v4-pro-0813": TokenPrice(
        input_usd_per_million=0.66,
        cached_input_usd_per_million=0.022,
        cache_write_usd_per_million=0.66,
        output_usd_per_million=1.98,
        pricing_source="https://api-docs.deepseek.com/quick_start/pricing",
        pricing_as_of="2026-08-23",
        estimate_note=(
            "Official DeepSeek-V4-Pro off-peak rates. Peak rates are 2x during "
            "01:00-04:00 and 06:00-10:00 UTC, Monday through Friday. Pass an "
            "observation timestamp for time-aware pricing."
        ),
    ),
}

_DEEPSEEK_V4_PRO_PEAK = TokenPrice(
    input_usd_per_million=1.32,
    cached_input_usd_per_million=0.044,
    cache_write_usd_per_million=1.32,
    output_usd_per_million=3.96,
    pricing_source="https://api-docs.deepseek.com/quick_start/pricing",
    pricing_as_of="2026-08-23",
    estimate_note=(
        "Official DeepSeek-V4-Pro peak rates for 01:00-04:00 and 06:00-10:00 "
        "UTC, Monday through Friday."
    ),
)


def _deepseek_peak(observed_at: str | datetime | None) -> bool:
    if observed_at is None:
        return False
    if isinstance(observed_at, datetime):
        timestamp = observed_at
    else:
        try:
            timestamp = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
        except ValueError:
            return False
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp.weekday() < 5 and (
        1 <= timestamp.hour < 4 or 6 <= timestamp.hour < 10
    )


def resolve_token_price(
    model: str | None,
    *,
    observed_at: str | datetime | None = None,
) -> tuple[str, TokenPrice] | None:
    if not model:
        return None
    normalized = model.lower().strip()
    # Claude Code appends this client-side suffix to advertise a one-million
    # token context window. The Anthropic transport strips it before sending
    # the provider model ID, so pricing must do the same.
    if normalized.endswith("[1m]"):
        normalized = normalized[:-4]
    for prefix in sorted(_PRICES, key=len, reverse=True):
        if normalized == prefix or normalized.startswith(prefix + "-"):
            if prefix == "deepseek-ai/deepseek-v4-pro-0813" and _deepseek_peak(
                observed_at
            ):
                return prefix, _DEEPSEEK_V4_PRO_PEAK
            return prefix, _PRICES[prefix]
    return None


def estimate_usage_cost(
    model: str | None,
    usage: dict[str, Any] | None,
    *,
    observed_at: str | datetime | None = None,
) -> dict[str, Any] | None:
    resolved = resolve_token_price(model, observed_at=observed_at)
    if resolved is None or usage is None:
        return None
    pricing_model, price = resolved
    input_tokens = int(usage.get("input_tokens") or 0)
    cached_tokens = min(input_tokens, int(usage.get("cached_input_tokens") or 0))
    uncached_tokens = max(0, input_tokens - cached_tokens)
    cache_write_tokens = min(
        uncached_tokens,
        int(usage.get("cache_write_input_tokens") or 0),
    )
    regular_uncached_tokens = uncached_tokens - cache_write_tokens
    cache_write_rate = (
        price.cache_write_usd_per_million
        if price.cache_write_usd_per_million is not None
        else price.input_usd_per_million
    )

    uncached_usd = regular_uncached_tokens * price.input_usd_per_million / 1_000_000
    cache_write_usd = cache_write_tokens * cache_write_rate / 1_000_000
    cached_usd = cached_tokens * price.cached_input_usd_per_million / 1_000_000
    output_tokens = int(usage.get("output_tokens") or 0)
    output_usd = output_tokens * price.output_usd_per_million / 1_000_000
    total_usd = uncached_usd + cache_write_usd + cached_usd + output_usd
    return {
        "currency": "USD",
        "pricing_model": pricing_model,
        "estimated_total_usd": round(total_usd, 9),
        "regular_uncached_input_usd": round(uncached_usd, 9),
        "cache_write_input_usd": round(cache_write_usd, 9),
        "cached_input_usd": round(cached_usd, 9),
        "output_usd": round(output_usd, 9),
        "rates_usd_per_million": {
            "input": price.input_usd_per_million,
            "cached_input": price.cached_input_usd_per_million,
            "cache_write_input": cache_write_rate,
            "output": price.output_usd_per_million,
        },
        "tokens": {
            "regular_uncached_input": regular_uncached_tokens,
            "cache_write_input": cache_write_tokens,
            "cached_input": cached_tokens,
            "output": output_tokens,
        },
        "pricing_source": price.pricing_source,
        "pricing_as_of": price.pricing_as_of,
        "estimate_note": price.estimate_note,
        "pricing_tier": (
            "peak"
            if pricing_model == "deepseek-ai/deepseek-v4-pro-0813"
            and _deepseek_peak(observed_at)
            else "off_peak"
            if pricing_model == "deepseek-ai/deepseek-v4-pro-0813"
            else None
        ),
    }
