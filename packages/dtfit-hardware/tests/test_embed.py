"""Guards for the embedded-LSI codegen -- the two facts the on-silicon numbers
rest on, both hardware-free:

* the float64 golden reproduces the real ``dtfit.streaming.LSIFilter`` (so the
  firmware is the dtfit method, not a lookalike);
* the checked-in flash tables match the generator, and every sketch dir carries
  identical copies (Arduino needs sketch-local headers), so a config change can
  never ship one sketch stale.
"""
from __future__ import annotations

from dtfit_hardware.tools import embed_lsi


def test_golden_matches_real_lsi_filter() -> None:
    # bit-faithful: the embedded float64 golden == the configured LSIFilter.
    assert embed_lsi.cross_check() < 1e-6


def test_checked_in_tables_match_generator() -> None:
    generated = embed_lsi.render_header()
    for target in embed_lsi.FIRMWARE_TARGETS:
        header = embed_lsi.FIRMWARE / target / "lsi_tables.h"
        assert header.is_file(), f"missing firmware/{target}/lsi_tables.h"
        assert header.read_text(encoding="utf-8") == generated, (
            f"firmware/{target}/lsi_tables.h is stale -- run "
            "`python -m dtfit_hardware.tools.embed_lsi` to regenerate"
        )


def test_shared_headers_are_in_sync_across_sketch_dirs() -> None:
    # dtfit_lsi.h is hand-written C copied into each sketch dir; a drift between
    # copies would ship one sketch with a different filter.
    dirs = [embed_lsi.FIRMWARE / t for t in embed_lsi.FIRMWARE_TARGETS]
    for fname in ("dtfit_lsi.h", "lsi_tables.h"):
        texts = {d.name: (d / fname).read_text(encoding="utf-8")
                 for d in dirs if (d / fname).is_file()}
        assert len(set(texts.values())) == 1, (
            f"{fname} differs across sketch dirs: {sorted(texts)}"
        )
