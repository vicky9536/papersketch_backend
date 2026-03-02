from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import tempfile
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class DownloadedPDF:
    path: str
    sha256: str
    size_bytes: int
    final_url: str


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
        )
    except ValueError:
        return True


def _hostname_resolves_to_private(hostname: str) -> bool:
    """
    Basic SSRF protection: block URLs that resolve to private/loopback IPs.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for info in infos:
        ip = info[4][0]
        if _is_private_ip(ip):
            return True
    return False


def _normalize_arxiv_pdf_url(url: str) -> str:
    """
    Accept:
      - https://arxiv.org/abs/XXXX.XXXXX
      - https://arxiv.org/pdf/XXXX.XXXXX
    Normalize to a direct PDF URL.
    """
    u = url.strip()
    if "arxiv.org/abs/" in u:
        u = u.replace("arxiv.org/abs/", "arxiv.org/pdf/")
    if "arxiv.org/pdf/" in u and not u.endswith(".pdf"):
        u = u + ".pdf"
    return u


def download_pdf(
    url: str,
    *,
    timeout_sec: int = 60,
    max_size_mb: int = 50,
    user_agent: Optional[str] = None,
) -> DownloadedPDF:
    """
    Download a PDF from a public URL into a temp file.

    Security:
      - Only http/https
      - Blocks private/loopback IP resolution (basic SSRF mitigation)
      - Enforces max download size

    Returns:
      DownloadedPDF(path, sha256, size_bytes, final_url)
    """
    if not url or not isinstance(url, str):
        raise ValueError("url is required")

    url = _normalize_arxiv_pdf_url(url)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")

    if not parsed.hostname:
        raise ValueError("URL hostname missing")

    # SSRF mitigation: block private networks
    if _hostname_resolves_to_private(parsed.hostname):
        raise ValueError("Blocked URL hostname (resolves to private/loopback IP)")

    headers = {
        "User-Agent": user_agent or "PaperSketchBackend/0.1 (+https://example.invalid)",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    max_bytes = max_size_mb * 1024 * 1024

    with requests.get(url, headers=headers, stream=True, timeout=timeout_sec, allow_redirects=True) as r:
        r.raise_for_status()

        # Some servers don't set content-type properly; we allow if bytes look like PDF later.
        content_type = (r.headers.get("Content-Type") or "").lower()

        sha = hashlib.sha256()
        size = 0

        fd, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"PDF too large (>{max_size_mb} MB limit)")
                    sha.update(chunk)
                    f.write(chunk)

            # quick signature check
            with open(path, "rb") as f:
                sig = f.read(4)
            if sig != b"%PDF":
                # allow rare cases where server sends something else
                raise ValueError("Downloaded content is not a PDF (missing %PDF signature)")

            return DownloadedPDF(
                path=path,
                sha256=sha.hexdigest(),
                size_bytes=size,
                final_url=str(r.url),
            )
        except Exception:
            # cleanup on failure
            try:
                os.remove(path)
            except OSError:
                pass
            raise
