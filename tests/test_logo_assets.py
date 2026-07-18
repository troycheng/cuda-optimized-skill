from __future__ import annotations

import struct
import unittest
import xml.etree.ElementTree as ET
import zlib
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


def paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    distances = (abs(estimate - left), abs(estimate - up), abs(estimate - upper_left))
    return (left, up, upper_left)[distances.index(min(distances))]


def read_png(name: str) -> tuple[int, int, int, bytes]:
    content = (ASSET_DIR / name).read_bytes()
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{name} does not have a PNG signature")
    offset = 8
    idat = bytearray()
    width = height = color_type = None
    while offset < len(content):
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        chunk_type = content[offset + 4 : offset + 8]
        data = content[offset + 8 : offset + 8 + length]
        offset += length + 12
        if chunk_type == b"IHDR":
            width, height, depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            if (depth, color_type, compression, filtering, interlace) != (8, 6, 0, 0, 0):
                raise AssertionError(f"{name} is not an 8-bit non-interlaced RGBA PNG")
        elif chunk_type == b"IDAT":
            idat.extend(data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError(f"{name} does not contain IHDR")

    raw = zlib.decompress(idat)
    stride = width * 4
    previous = bytearray(stride)
    alpha = bytearray()
    position = 0
    for _row in range(height):
        filter_type = raw[position]
        position += 1
        scanline = bytearray(raw[position : position + stride])
        position += stride
        for index, value in enumerate(scanline):
            left = scanline[index - 4] if index >= 4 else 0
            up = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 1:
                scanline[index] = (value + left) & 0xFF
            elif filter_type == 2:
                scanline[index] = (value + up) & 0xFF
            elif filter_type == 3:
                scanline[index] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                scanline[index] = (value + paeth(left, up, upper_left)) & 0xFF
            elif filter_type != 0:
                raise AssertionError(f"{name} uses unknown PNG filter {filter_type}")
        alpha.extend(scanline[3::4])
        previous = scanline
    return width, height, color_type, bytes(alpha)


def expected_geometry() -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    result = [("title", (("id", "title"),))]
    for y in (7, 36, 65):
        for x in (7, 36, 65):
            if (x, y) == (36, 36):
                continue
            result.append(
                (
                    "rect",
                    tuple(sorted({"x": str(x), "y": str(y), "width": "24", "height": "24", "rx": "5"}.items())),
                )
            )
    result.append(("path", (("d", "M34 48 48 34 62 48 48 62Z"),)))
    return tuple(result)


class LogoAssetTests(unittest.TestCase):
    def test_logo_asset_contract(self) -> None:
        light = parse_svg("logo.svg")
        dark = parse_svg("logo-dark.svg")

        self.assertEqual(light.attrib["viewBox"], "0 0 96 96")
        self.assertEqual(dark.attrib["viewBox"], "0 0 96 96")
        self.assertNotIn("width", light.attrib)
        self.assertNotIn("height", light.attrib)
        self.assertNotIn("width", dark.attrib)
        self.assertNotIn("height", dark.attrib)
        self.assertEqual(geometry(light), geometry(dark))
        self.assertEqual(geometry(light), expected_geometry())
        self.assertEqual(fills(light), Counter({"#172033": 8, "#16B8A6": 1}))
        self.assertEqual(fills(dark), Counter({"#F5F7FA": 8, "#28D6C2": 1}))
        for name, size in (("logo-128.png", 128), ("logo-512.png", 512)):
            width, height, color_type, alpha = read_png(name)
            self.assertEqual((width, height, color_type), (size, size, 6))
            self.assertEqual(min(alpha), 0)
            self.assertEqual(max(alpha), 255)

        for readme_name in ("README.md", "README.zh-CN.md"):
            readme = (ROOT / readme_name).read_text(encoding="utf-8")
            self.assertIn(
                '<source media="(prefers-color-scheme: dark)" srcset="asset/logo-dark.svg">',
                readme,
            )
            self.assertIn('<img src="asset/logo.svg" width="88"', readme)

    def test_wordmark_asset_contract(self) -> None:
        expected = {
            "logo-wordmark.svg": ("#172033", "#16B8A6"),
            "logo-wordmark-dark.svg": ("#F5F7FA", "#28D6C2"),
        }
        for name, (foreground, accent) in expected.items():
            root = parse_svg(name)
            self.assertEqual(root.attrib["viewBox"], "0 0 720 152")
            self.assertNotIn("width", root.attrib)
            self.assertNotIn("height", root.attrib)
            self.assertEqual(root.attrib["role"], "img")
            self.assertEqual(root.attrib["aria-labelledby"], "title")

            titles = [
                element
                for element in root.iter()
                if local_name(element.tag) == "title"
            ]
            self.assertEqual(len(titles), 1)
            self.assertIn("CUDA Kernel Optimizer", titles[0].text or "")

            full_backgrounds = [
                element
                for element in root.iter()
                if local_name(element.tag) == "rect"
                and element.attrib.get("width") == "720"
                and element.attrib.get("height") == "152"
            ]
            self.assertEqual(full_backgrounds, [])

            icon_groups = [
                element
                for element in root.iter()
                if element.attrib.get("data-role") == "thread-tile"
            ]
            self.assertEqual(len(icon_groups), 1)
            self.assertEqual(geometry(icon_groups[0]), expected_geometry()[1:])

            labels = [
                " ".join((element.text or "").split())
                for element in root.iter()
                if local_name(element.tag) == "text"
            ]
            self.assertEqual(labels, ["CUDA KERNEL", "OPTIMIZER"])
            source = (ASSET_DIR / name).read_text(encoding="utf-8")
            self.assertIn(foreground, source)
            self.assertIn(accent, source)
            self.assertIn("ui-sans-serif", source)


if __name__ == "__main__":
    unittest.main()
