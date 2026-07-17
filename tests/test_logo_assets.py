from __future__ import annotations

import struct
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "asset"


def parse_svg(name: str) -> ET.Element:
    return ET.parse(ASSET_DIR / name).getroot()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def geometry(root: ET.Element) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    return tuple(
        (
            local_name(element.tag),
            tuple(sorted((key, value) for key, value in element.attrib.items() if key != "fill")),
        )
        for element in root.iter()
        if element is not root
    )


def fills(root: ET.Element) -> Counter[str]:
    return Counter(
        element.attrib["fill"]
        for element in root.iter()
        if element is not root and "fill" in element.attrib
    )


def read_png_ihdr(name: str) -> tuple[int, int, int]:
    with (ASSET_DIR / name).open("rb") as file:
        signature = file.read(8)
        length = struct.unpack(">I", file.read(4))[0]
        chunk_type = file.read(4)
        data = file.read(length)
    if signature != b"\x89PNG\r\n\x1a\n" or chunk_type != b"IHDR" or length != 13:
        raise AssertionError(f"{name} does not start with a valid PNG IHDR")
    width, height, _depth, color_type, _compression, _filter, _interlace = struct.unpack(
        ">IIBBBBB", data
    )
    return width, height, color_type


class LogoAssetTests(unittest.TestCase):
    def test_logo_asset_contract(self) -> None:
        light = parse_svg("logo.svg")
        dark = parse_svg("logo-dark.svg")

        self.assertEqual(light.attrib["viewBox"], "0 0 96 96")
        self.assertEqual(dark.attrib["viewBox"], "0 0 96 96")
        self.assertEqual(geometry(light), geometry(dark))
        self.assertEqual(fills(light), Counter({"#172033": 8, "#16B8A6": 1}))
        self.assertEqual(fills(dark), Counter({"#F5F7FA": 8, "#28D6C2": 1}))
        self.assertEqual(read_png_ihdr("logo-128.png"), (128, 128, 6))
        self.assertEqual(read_png_ihdr("logo-512.png"), (512, 512, 6))

        for readme_name in ("README.md", "README.zh-CN.md"):
            readme = (ROOT / readme_name).read_text(encoding="utf-8")
            self.assertIn('<img src="asset/logo.svg" width="88"', readme)


if __name__ == "__main__":
    unittest.main()
