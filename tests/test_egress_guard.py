"""Canary proving the egress_guard fixture blocks outbound connections.

The guard lives in conftest.py. These tests exercise it: a connect to a
non-allowlisted host must raise EgressBlockedError, and an allowlisted host
must pass the guard's check (delegating to the real connect). The canary uses
pytest.raises internally, so it PASSES when the guard fires.
"""

import socket

import pytest

from tests.conftest import EGRESS_ALLOWLIST, EgressBlockedError


def test_egress_canary_blocks_outbound(egress_guard: None) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(EgressBlockedError):
            sock.connect(("example.com", 80))
    finally:
        sock.close()


def test_egress_canary_allows_loopback_host(egress_guard: None) -> None:
    # An allowlisted host must not raise EgressBlockedError. The guard delegates
    # to the real connect, which may still fail for its own reasons (port closed,
    # or the outer --block-network guard). Only EgressBlockedError is a guard
    # failure here.
    assert "127.0.0.1" in EGRESS_ALLOWLIST
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", 9))
    except EgressBlockedError:
        pytest.fail("egress_guard blocked an allowlisted host")
    except (OSError, RuntimeError):
        pass
    finally:
        sock.close()
