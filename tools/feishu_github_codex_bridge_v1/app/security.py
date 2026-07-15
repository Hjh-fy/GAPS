import hashlib
import hmac
import time
from typing import Mapping, Optional


def verify_feishu_signature(
    *,
    raw_body: bytes,
    headers: Mapping[str, str],
    encrypt_key: str,
    max_age_seconds: Optional[int] = 300,
) -> bool:
    """Verify Feishu/Lark event callback signature.

    Feishu webhook signature is:
        sha256(timestamp + nonce + encrypt_key + raw_body)

    This function intentionally reads the raw request body before JSON parsing.
    """
    if not encrypt_key:
        return True

    lowered = {str(k).lower(): str(v) for k, v in headers.items()}
    signature = lowered.get("x-lark-signature", "")
    timestamp = lowered.get("x-lark-request-timestamp", "")
    nonce = lowered.get("x-lark-request-nonce", "")

    if not signature or not timestamp or not nonce:
        return False

    if max_age_seconds is not None:
        try:
            ts = int(timestamp)
        except ValueError:
            return False
        if abs(int(time.time()) - ts) > max_age_seconds:
            return False

    content = (timestamp + nonce + encrypt_key).encode("utf-8") + raw_body
    expected = hashlib.sha256(content).hexdigest()
    return hmac.compare_digest(expected, signature)
