"""The vendored code and frozen data are byte-identical to phishing-url-detector@4283da1.

The study's validity depends on using the original's canonicalisation, Public
Suffix List, features and dataset assembly unchanged. An edit to any of them —
even a well-meant one — would quietly change what is being compared, so their
hashes are pinned here.
"""

from __future__ import annotations

import hashlib

import pytest

from phishsmall.config import ROOT

ORIGINAL_SHA256 = {
    "src/phishurl/__init__.py": "9b31aa6a37112edd11c469f7646d0b889d7c1722138d2131e0c315a0a9bc4759",
    "src/phishurl/canonicalize.py": "90c1198a8ae1a0678c1e23e3ce845309ee84ac2a08f116f37c3804bb900a5b43",
    "src/phishurl/psl.py": "c8689d818d5c66cec8f107af5ef92ce2666f6547e373bea228e7e9295713b85d",
    "src/phishurl/features.py": "d19e68b1e3a294f5077e1835ec30cc757ddbd57fe8c725536630d7658c0bb33a",
    "src/phishurl/datasets.py": "1696bfeafebc8fd1cc4e7f2cd7f7d39758b8c81444539ce201c1019f792dfdd5",
    "src/phishurl/evaluate.py": "d121ef3df012f605b1db1d3abc9de78b4710554adf736be11394786a8cc7f421",
    "src/phishurl/viz.py": "3f7a241c08fe9679a94a36e5e4cd80db6c7067c2fd4916be70ffa9d738ac3356",
    "src/phishurl/train.py": "b38b90e84a39324ec557dc183ad7212b0e9cb262b36317abddadfafe3930333b",
    "data/public_suffix_list.dat": "084a5674d77c1d14900b16da5fc8afee9765af2f00a638552a8c7aa18f44ae81",
    "data/brand_tokens.txt": "e6c193e03c0181ace198ff45ffc181551428b5198f811e236ecfc523b1d4ebc4",
}


@pytest.mark.parametrize("relative_path", sorted(ORIGINAL_SHA256))
def test_vendored_file_is_unchanged(relative_path: str) -> None:
    digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
    assert digest == ORIGINAL_SHA256[relative_path], f"{relative_path} differs from the original"
