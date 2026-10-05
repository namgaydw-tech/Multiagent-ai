#!/usr/bin/env python
"""Range-based remote ZIP reader for Phase 5 dataset acquisition.

The Kermany Mendeley archive (rscbjbr9sj, ZhangLabData.zip, 8.44 GB) contains
OCT + chest X-ray images. We only need the chest X-ray subset, so instead of
downloading 8.44 GB we read the ZIP central directory over HTTP range requests
and fetch only the byte ranges of the entries we ask for.

Usage:
    python scripts/_phase5_remote_zip.py list  <url> [--prefix P]
    python scripts/_phase5_remote_zip.py extract <url> --prefix P --dest DIR [--limit N] [--workers K]

Research prototype — not a medical device.
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib.request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}

EOCD_SIG = b"PK\x05\x06"
CDIR_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"


class RemoteZip:
    def __init__(self, url: str, timeout: int = 60):
        self.url, self.size = self._resolve(url)
        self.timeout = timeout
        self.entries = self._read_central_directory()

    @staticmethod
    def _resolve(url: str, timeout: int = 60) -> tuple[str, int]:
        """Follow redirects with a 1-byte GET; return final URL + total size."""
        req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final = resp.geturl()
            crange = resp.headers.get("Content-Range", "")
            total = int(crange.split("/")[1]) if "/" in crange else int(
                resp.headers.get("Content-Length", 0)
            )
        return final, total

    def _get(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(
            self.url, headers={**HEADERS, "Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        expected = end - start + 1
        if len(data) != expected:
            raise IOError(f"short read {len(data)} != {expected} for {start}-{end}")
        return data

    def _read_central_directory(self) -> list[dict]:
        # EOCD is at most 65555 + 22 bytes from the end.
        tail_len = min(self.size, 65555 + 22)
        tail = self._get(self.size - tail_len, self.size - 1)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise IOError("EOCD not found")
        (_sig, _disk, _cd_disk, _n_disk, _n_total, cd_size, cd_off, _clen) = struct.unpack(
            "<4s4H2LH", tail[i : i + 22]
        )
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or _n_total == 0xFFFF:
            # ZIP64: locator sits immediately before the EOCD record.
            j = i - 20
            if tail[j : j + 4] != b"PK\x06\x07":
                raise IOError("ZIP64 locator not found")
            (_s, _d, z64off, _disks) = struct.unpack("<4sLQL", tail[j : i])
            z = self._get(z64off, z64off + 55)
            if z[:4] != b"PK\x06\x06":
                raise IOError("ZIP64 EOCD not found")
            (_s2, _sz, _vm, _vn, _d1, _d2, _n_disk64, n_total64,
             cd_size64, cd_off64) = struct.unpack("<4sQ2H2L4Q", z[:56])
            cd_size, cd_off, _n_total = cd_size64, cd_off64, n_total64
        cd = self._get(cd_off, cd_off + cd_size - 1)
        entries = []
        p = 0
        while p < len(cd):
            if cd[p : p + 4] != CDIR_SIG:
                break
            (
                _sig, _ver_made, _ver_need, _flags, _method, _time, _date,
                _crc, csize, usize, namelen, extralen, commentlen,
                _diskno, _iattr, _eattr, local_off,
            ) = struct.unpack("<4s6H3L5H2L", cd[p : p + 46])
            name = cd[p + 46 : p + 46 + namelen].decode("utf-8", "replace")
            entries.append(
                {"name": name, "method": _method, "csize": csize,
                 "usize": usize, "local_off": local_off}
            )
            p += 46 + namelen + extralen + commentlen
        return entries

    def extract(self, entry: dict) -> bytes:
        off = entry["local_off"]
        hdr = self._get(off, off + 29)
        if hdr[:4] != LOCAL_SIG:
            raise IOError(f"bad local header for {entry['name']}")
        namelen, extralen = struct.unpack("<HH", hdr[26:30])
        data_start = off + 30 + namelen + extralen
        raw = self._get(data_start, data_start + entry["csize"] - 1)
        if entry["method"] == 0:
            return raw
        if entry["method"] == 8:
            return zlib.decompress(raw, -15)
        raise IOError(f"unsupported method {entry['method']}")


def coalesce(entries: list[dict], rz: RemoteZip, max_gap: int = 4096,
             max_chunk: int = 32 * 1048576) -> list[tuple[int, int, list[dict]]]:
    """Group entries into contiguous byte ranges (zip entries are stored in order)."""
    spans = []
    for e in sorted(entries, key=lambda e: e["local_off"]):
        hdr_len = 30 + len(e["name"].encode()) + 512  # upper bound for name+extra
        start = e["local_off"]
        end = start + hdr_len + e["csize"] + 1024
        spans.append((start, end, e))
    groups: list[tuple[int, int, list[dict]]] = []
    for start, end, e in spans:
        fits = (groups and start - groups[-1][1] <= max_gap
                and end - groups[-1][0] <= max_chunk)
        if fits:
            s, g, lst = groups[-1]
            groups[-1] = (s, max(g, end), lst + [e])
        else:
            groups.append((start, end, [e]))
    return groups


def extract_group(rz: RemoteZip, group: tuple[int, int, list[dict]], dest: Path) -> int:
    start, end, entries = group
    blob = rz._get(start, end - 1)
    for e in entries:
        rel = e["local_off"] - start
        if blob[rel : rel + 4] != LOCAL_SIG:
            raise IOError(f"bad local header for {e['name']}")
        namelen, extralen = struct.unpack("<HH", blob[rel + 26 : rel + 30])
        ds = rel + 30 + namelen + extralen
        raw = blob[ds : ds + e["csize"]]
        if e["method"] == 8:
            data = zlib.decompress(raw, -15)
        elif e["method"] == 0:
            data = raw
        else:
            raise IOError(f"unsupported method {e['method']}")
        out = dest / e["name"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return len(entries)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["list", "extract"])
    ap.add_argument("url")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rz = RemoteZip(args.url)
    sel = [e for e in rz.entries if e["name"].startswith(args.prefix) and not e["name"].endswith("/")]
    total = sum(e["usize"] for e in sel)
    print(f"archive={rz.size} entries={len(rz.entries)} selected={len(sel)} uncompressed={total}")

    if args.action == "list":
        for e in sel[:50]:
            print(f"  {e['usize']:>10} {e['name']}")
        if len(sel) > 50:
            print(f"  ... {len(sel) - 50} more")
        return 0

    if args.limit:
        sel = sel[: args.limit]
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    groups = coalesce(sel, rz)
    total_bytes = sum(g[1] - g[0] for g in groups)
    print(f"groups={len(groups)} bytes_to_fetch={total_bytes // 1048576} MB", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n in pool.map(lambda g: extract_group(rz, g, dest), groups):
            done += n
            if done % 500 < n or done == len(sel):
                print(f"  {done}/{len(sel)}", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
