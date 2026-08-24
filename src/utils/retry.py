"""Provider-agnostic retry helpers for quota-limited lab API calls."""

import re
import time


def retry_delay(exc: Exception, fallback: float) -> float:
    """Read a provider retry hint, falling back to bounded exponential delay."""
    message = str(exc)
    match = re.search(r"retry(?:Delay| in)[^0-9]*(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    if match:
        return max(float(match.group(1)) + 1.0, fallback)
    return fallback


def invoke_with_retry(
    operation,
    attempts: int = 20,
    base_delay: float = 2.0,
    max_delay: float = 65.0,
    sleep=time.sleep,
    label: str = "API request",
):
    """Invoke an operation with bounded backoff and provider retry hints."""
    last_error = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            fallback = min(base_delay * (2 ** attempt), max_delay)
            delay = min(retry_delay(exc, fallback), max_delay)
            print(f"⏳ {label} bị giới hạn; thử lại sau {delay:.1f}s ...")
            sleep(delay)
    raise last_error
