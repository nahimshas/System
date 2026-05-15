"""
Shared HTTP client for all data-fetching modules.

Centralizes timeout defaults, retry logic, and error logging so each
sport stats file doesn't need its own copy. Data files should call
get_json() instead of requests.get() directly.

Migration: data files are migrated to use this module during Phase 2
(sport module creation). Until then, they continue using requests directly.
"""

import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Default timeouts (seconds). Individual callers can override via the timeout param.
_DEFAULT_TIMEOUT = 12
_DEFAULT_RETRIES = 2
_RETRY_DELAY = 1.0   # seconds between retries


def get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
    default: Any = None,
    label: str = "",
) -> Any:
    """
    Fetch a URL and return the parsed JSON response.

    Args:
        url:     Full URL to fetch.
        params:  Optional query parameters dict.
        timeout: Request timeout in seconds.
        retries: Number of additional attempts on connection/timeout errors.
                 Does NOT retry on 4xx/5xx responses (those are not transient).
        default: Value to return on any failure (default: None).
        label:   Short description logged on failure (e.g. "NBA stats").

    Returns:
        Parsed JSON (dict or list) on success, `default` on any failure.
    """
    tag = f"[{label}] " if label else ""
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < attempts:
                logger.warning(f"{tag}Timeout (attempt {attempt}/{attempts}), retrying...")
                time.sleep(_RETRY_DELAY)
            else:
                logger.error(f"{tag}Timeout after {attempts} attempt(s): {url}")
        except requests.exceptions.ConnectionError:
            if attempt < attempts:
                logger.warning(f"{tag}Connection error (attempt {attempt}/{attempts}), retrying...")
                time.sleep(_RETRY_DELAY)
            else:
                logger.error(f"{tag}Connection error after {attempts} attempt(s): {url}")
        except requests.exceptions.HTTPError as e:
            # HTTP errors (4xx/5xx) are not transient — don't retry
            logger.error(f"{tag}HTTP {e.response.status_code}: {url}")
            break
        except Exception as e:
            logger.error(f"{tag}Unexpected error fetching {url}: {e}")
            break

    return default
