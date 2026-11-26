"""Render the public Hugging Face lifetime download count into the README badge."""

import json
from pathlib import Path
from urllib.request import urlopen

API = "https://huggingface.co/api/datasets/hding2455/TACTUS?expand=downloadsAllTime"
BADGE = Path(__file__).resolve().parents[1] / "assets" / "dataset-downloads.svg"

with urlopen(API, timeout=30) as response:
    count = json.load(response)["downloadsAllTime"]
if type(count) is not int or count < 0:
    raise ValueError("Expected a nonnegative integer in downloadsAllTime")

value = f"{count:,}"
label_width = 132
value_width = 20 + 9 * len(value)
width = label_width + value_width
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="26" viewBox="0 0 {width} 26" role="img" aria-labelledby="title">
  <title id="title">Total dataset downloads: {value}</title>
  <rect width="{width}" height="26" rx="4" fill="#984000"/>
  <path fill="#202020" d="M4 0h{label_width - 4}v26H4a4 4 0 0 1-4-4V4a4 4 0 0 1 4-4z"/>
  <g fill="none" stroke="#ffbd83" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M13 6v9m-3-3 3 3 3-3M8 16v3h10v-3"/>
  </g>
  <g fill="#fff" font-family="Verdana,Arial,sans-serif" font-size="12">
    <text x="25" y="17">Total downloads</text>
    <text x="{label_width + value_width / 2:g}" y="17" text-anchor="middle" font-weight="bold">{value}</text>
  </g>
</svg>
"""
with BADGE.open("w", encoding="utf-8", newline="\n") as output:
    output.write(svg)
