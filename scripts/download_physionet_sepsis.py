#!/usr/bin/env python3
"""Download the public PhysioNet 2019 challenge training records from its S3 mirror."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "challenge-2019/1.0.0/training/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
THREAD_STATE = threading.local()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=.4, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32))
    session.headers.update({"User-Agent": "PlumoRingAI-research/1.0 (public dataset download)"})
    return session


def list_records(session: requests.Session, sets: list[str]) -> list[str]:
    keys: list[str] = []
    for set_name in sets:
        token = None
        prefix = f"{PREFIX}{set_name}/"
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                query["continuation-token"] = token
            response = session.get(BASE + "?" + urlencode(query), timeout=45)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            keys.extend(node.text for node in root.findall(".//s3:Contents/s3:Key", NS) if node.text and node.text.endswith(".psv"))
            if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
                break
            token = root.findtext("s3:NextContinuationToken", namespaces=NS)
            if not token:
                raise RuntimeError("S3 listing was truncated without a continuation token.")
    return keys


def fetch_one(args: tuple[str, str, Path]) -> tuple[str, bool]:
    key, base, destination = args
    path = destination / key.removeprefix(PREFIX)
    if path.is_file() and path.stat().st_size > 100:
        return key, False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    session = getattr(THREAD_STATE, "session", None)
    if session is None:
        session = make_session()
        THREAD_STATE.session = session
    response = session.get(base + key, timeout=(15, 90), stream=True)
    response.raise_for_status()
    with temporary.open("wb") as file:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                file.write(chunk)
    if temporary.stat().st_size < 100:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete public dataset file: {key}")
    os.replace(temporary, path)
    return key, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/physionet_sepsis_2019"))
    parser.add_argument("--sets", nargs="+", default=["training_setA", "training_setB"], choices=["training_setA", "training_setB"])
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = make_session()
    keys = list_records(session, args.sets)
    if not keys:
        raise RuntimeError("No .psv records found in the official public dataset mirror.")
    print(f"Found {len(keys):,} public patient files; downloading with {args.workers} workers.", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_one, (key, BASE, args.output)) for key in keys]
        done = 0
        fetched = 0
        for future in concurrent.futures.as_completed(futures):
            _, downloaded = future.result()
            fetched += int(downloaded)
            done += 1
            if done % 1000 == 0 or done == len(futures):
                print(f"{done:,}/{len(futures):,} files ready ({fetched:,} downloaded this run).", flush=True)
    print(f"Dataset files are in {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
