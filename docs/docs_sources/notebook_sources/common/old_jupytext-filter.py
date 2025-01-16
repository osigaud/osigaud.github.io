# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Process percent notebooks
"""

(1) Replace

# [[student]] Instructions 
# [[assert]] Other assert instruction
...
## hint = 2
# [[/student]]

by

# Instructions
assert False, "Code non implémenté"
hint = 2

(2) Removes [[remove]] ... [[/remove]] sections

(3) Inline images
"""

import base64
import json
import argparse
import mimetypes
from pathlib import Path
import re
import sys
import logging
logging.basicConfig(level=logging.INFO)
import jupytext

re_student_start = re.compile(r"""^(\s*)#(.*)\[\[STUDENT\]\]\s*(\S.*)?""", re.IGNORECASE)
re_student_end = re.compile(r""".*\[\[/STUDENT\]\]""", re.IGNORECASE)
re_hint = re.compile(r"""(.*)##(?:\s*)(\S.*)$""", re.IGNORECASE)
re_assert = re.compile(r"""^(\s*)#(?:.*)\[\[assert\]\]\s*(\S.*)$""", re.IGNORECASE)

re_remove_start = re.compile(r""".*\[\[REMOVE\]\]""", re.IGNORECASE)
re_remove_end = re.compile(r""".*\[\[/REMOVE\]\]""", re.IGNORECASE)

RE_MARKDOWN_IMAGE = re.compile(r"""!\[([^]]+)\]\(([^\)]+)\)""")
RE_IMPORT_ALL = re.compile(r"""from (\w+) import \*\s*""")

def csv_list(string):
   return string.split(',')


def inline_image(matches):
    title = matches[1]
    path = matches[2]
    if path.startswith("http:") or path.startswith("https:"):
        return matches[0]
    if path.startswith("data:"):
        encoded = path
    else:
        path = Path(path)
        assert path.isfile(), "Cannot find {path}"

        mime_type, encoding = mimetypes.guess_type(path)
        image_b64 = base64.encodebytes(path.read_bytes()).decode("ascii").replace("\n", "")
        encoded = "data:{mime_type};base64,{image_b64}"
    
    return f"![{title}]({encoded})"

parser = argparse.ArgumentParser(description="Process Python (percent) notebooks to produce teacher/student / colab versions.")
parser.add_argument("--exclude", type = csv_list)
parser.add_argument("--include", type = csv_list)
parser.add_argument("--teacher", action='store_true', default=False)
parser.add_argument("source", nargs="?")




args = parser.parse_args()
exclude_tags = set(args.exclude or [])
included_tags = set(args.include or [])

teacher_mode = args.teacher


document = jupytext.read(open(args.source) if args.source else sys.stdin, fmt='py:percent')

def process(document):
    cells = []

    for ix, cell in enumerate(document["cells"]):
        lines = []
        hide = False
        remove = False
        
        tags = cell.get("metadata", {}).get("tags", [])
        cell_type = cell["cell_type"]

        if not any(tag in included_tags for tag in tags) and any(tag in exclude_tags for tag in tags):
            continue

        if "copy" in tags:
            for line in cell['source'].split('\n'):
                if m := RE_IMPORT_ALL.match(line):
                    module = m.group(1)
                    with Path(f"{module}.py").open("rt") as fp:
                        r = process(jupytext.read(fp, fmt='py:percent'))
                        cells.extend(r["cells"])
            continue
                        
        for lineno, line in enumerate(cell['source'].split('\n')):
            if m := re_student_start.match(line):
                # lines.append()
                lines.append(f"{m.group(1)}# À compléter... {m.group(2)}\n")
                assert not hide,  f"Pas de [[/student]] correspondant à un [[student]] dans la cellule {ix+1}: {cell['source'][:2]}"
                assert_ix = len(lines)
                if teacher_mode:
                    lines.append(f"""{m.group(1)}# assert False, 'Code non implémenté'\n""")
                else:
                    lines.append(f"""{m.group(1)}assert False, 'Code non implémenté'\n""")
                hide = lineno + 1
            elif re_student_end.match(line) is not None:
                assert hide,  f"Pas de [[student]] correspondant à un [[/student]] dans la cellule {ix+1}: {cell['source'][:2]}"
                hide = False
            elif m := re_assert.match(line):
                assert hide, f"Pas de [[student]] pour un [[assert]]"
                assert assert_ix, f"[[assert]] en double line {lineno}"
                lines[assert_ix] = None
                assert_ix =  None
                lines.append(f"""{m.group(1)}{m.group(2)}\n""")
            elif re_remove_start.match(line):
                assert not remove, "No [[/remove]] tag"
                remove = True
            elif re_remove_end.match(line):
                assert remove, "No [[remove]] tag for this [[/remove]]"
                remove = False
            elif hide and re_hint.match(line) is not None:
                lines.append(re_hint.sub(r"\1\2", line))
            elif not (hide or remove) or teacher_mode:
                if cell_type == "markdown":
                    line = RE_MARKDOWN_IMAGE.sub(inline_image, line)
                lines.append(line)

        assert not hide, f"Pas de [[/student]] correspondant à un [[student]] dans la cellule {ix+1}: {cell['source'][:2]}"
        assert not remove, f"Pas de [[/remove]] correspondant à un [[remove]] dans la cellule {ix+1}: {cell['source'][:2]}"

        # Change source
        cell['source'] = "\n".join([line for line in lines if line is not None])

        # Remove outputs
        if "outputs" in cell:
            cell["outputs"] = []

        # Remove colab output
        if metadata := cell.get("metadata", None):
            if "colab" in metadata:
                del metadata["colab"]

        cells.append(cell)

    document["cells"] = cells
    return document

document = process(document)
jupytext.write(document, sys.stdout, fmt='ipynb')
