#!/usr/bin/env python
"""Convert the two proposal markdown files into formatted Word documents.

Deliberately a targeted converter rather than a general one: it handles exactly
the constructs these two documents use (headings, paragraphs with bold/code
spans, pipe tables, fenced code blocks, block quotes, bullet and numbered lists,
horizontal rules) and applies formal Chinese/English document styling.

Chinese text needs an explicit ``w:eastAsia`` font in addition to
``run.font.name``; setting only the Latin font leaves CJK glyphs to whatever the
reader substitutes.

Usage:
    <bundled python> docs/proposal/md2docx.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent

LATIN_BODY = "Times New Roman"
CJK_BODY = "宋体"
LATIN_HEAD = "Arial"
CJK_HEAD = "黑体"
MONO = "Consolas"

BODY_PT = 12
LINE_SPACING = 1.5

# Chinese convention indents two characters (about 0.74 cm at 12 pt) and keeps a
# small space between paragraphs. English convention indents half an inch and
# does NOT add paragraph spacing on top, because the indent alone marks the break.
INDENT_ZH = 0.74
INDENT_EN = 1.27
REF_HANG = 0.75


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def set_run_fonts(run, latin: str, cjk: str) -> None:
    """Set both the Latin and the East Asian font of a run."""
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), cjk)
    rfonts.set(qn("w:cs"), latin)


def shade(paragraph, hex_fill: str) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    ppr.append(shd)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    for el, attr in (("w:fldChar", {"w:fldCharType": "begin"}),
                     ("w:instrText", None),
                     ("w:fldChar", {"w:fldCharType": "end"})):
        e = OxmlElement(el)
        if attr:
            for k, v in attr.items():
                e.set(qn(k), v)
        else:
            e.set(qn("xml:space"), "preserve")
            e.text = " PAGE "
        run._r.append(e)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def smart_quotes(text: str) -> str:
    """Turn straight double quotes into typographic pairs.

    Markdown sources carry ASCII quotes, which look unprofessional in a submitted
    document. Quotes are alternated within a paragraph, which is correct because
    every paragraph here is passed in one call.
    """
    out = []
    opening = True
    for ch in text:
        if ch == '"':
            out.append("\u201c" if opening else "\u201d")
            opening = not opening
        else:
            out.append(ch)
    return "".join(out)


def add_rich_text(paragraph, text: str, *, latin: str, cjk: str, base_bold=False) -> None:
    """Add text to a paragraph, honouring **bold** and `code` spans."""
    text = smart_quotes(text)
    for chunk in INLINE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
            set_run_fonts(run, latin, cjk)
        elif chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 2:
            run = paragraph.add_run(chunk[1:-1])
            set_run_fonts(run, MONO, cjk)
            run.font.size = Pt(BODY_PT - 1)
        else:
            run = paragraph.add_run(chunk)
            if base_bold:
                run.bold = True
            set_run_fonts(run, latin, cjk)


# --------------------------------------------------------------------------
# document styling
# --------------------------------------------------------------------------
def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = LATIN_BODY
    normal.font.size = Pt(BODY_PT)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), LATIN_BODY)
    rfonts.set(qn("w:hAnsi"), LATIN_BODY)
    rfonts.set(qn("w:eastAsia"), CJK_BODY)

    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = LINE_SPACING
    pf.space_after = Pt(6)

    for name, size, color in (("Title", 22, RGBColor(0, 0, 0)),
                              ("Heading 1", 16, RGBColor(0, 0, 0)),
                              ("Heading 2", 13, RGBColor(0x1A, 0x1A, 0x1A)),
                              ("Heading 3", 11.5, RGBColor(0x1A, 0x1A, 0x1A))):
        st = doc.styles[name]
        st.font.name = LATIN_HEAD
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        rp = st.element.get_or_add_rPr()
        rf = rp.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rp.append(rf)
        rf.set(qn("w:ascii"), LATIN_HEAD)
        rf.set(qn("w:hAnsi"), LATIN_HEAD)
        rf.set(qn("w:eastAsia"), CJK_HEAD)
        st.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
        st.paragraph_format.space_after = Pt(8)
        st.paragraph_format.keep_with_next = True


def configure_page(doc: Document) -> None:
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(p)
    for run in p.runs:
        set_run_fonts(run, LATIN_BODY, CJK_BODY)
        run.font.size = Pt(9)


# --------------------------------------------------------------------------
# markdown parsing
# --------------------------------------------------------------------------
def split_table_row(line: str) -> list:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def is_separator_row(cells: list) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c) and any(cells)


def add_table(doc: Document, rows: list) -> None:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    table = doc.add_table(rows=0, cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Column widths: proportional to the longest cell, but water-filled so the
    # row total lands exactly on the text width. Clamping to a minimum and then
    # scaling would push the total past the margin, which is what happened on a
    # first attempt (a 3-column table reached 17.5 cm against a 16.0 cm text
    # width and ran off the page).
    usable = 16.0
    min_w = 1.6

    def disp_len(text: str) -> int:
        """Approximate rendered width, counting CJK glyphs as two units.

        Plain len() counts a Chinese character the same as a Latin letter, which
        under-weights short Chinese columns; "决策保真" then wrapped to two lines
        while a long English cell next to it kept its width.
        """
        w = 0
        for ch in text:
            w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        return w

    longest = []
    for c in range(ncols):
        lens = [disp_len(r[c]) if c < len(r) else 0 for r in rows]
        longest.append(max(1, max(lens)))

    if ncols * min_w >= usable:
        widths = [usable / ncols] * ncols
    else:
        widths = [0.0] * ncols
        frozen = set()
        remaining = usable
        for _ in range(ncols):
            free = [c for c in range(ncols) if c not in frozen]
            if not free:
                break
            share = remaining / sum(longest[c] for c in free)
            for c in free:
                widths[c] = share * longest[c]
            below = [c for c in free if widths[c] < min_w]
            if not below:
                break
            for c in below:
                frozen.add(c)
                widths[c] = min_w
                remaining -= min_w
        for c in range(ncols):
            if c not in frozen:
                pass
        scale = usable / sum(widths)
        widths = [w * scale for w in widths]

    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)

    for ri, row in enumerate([header] + body):
        cells = table.add_row().cells
        for ci in range(ncols):
            text = row[ci] if c < len(row) else ""
            cell = cells[ci]
            cell.width = Cm(widths[ci])
            para = cell.paragraphs[0]
            para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            para.paragraph_format.space_after = Pt(2)
            add_rich_text(para, text, latin=LATIN_BODY, cjk=CJK_BODY,
                          base_bold=(ri == 0))
            if ri == 0:
                for run in para.runs:
                    run.bold = True
                shade(para, "E8EDF3")

    for ci, col in enumerate(table.columns):
        col.width = Cm(widths[ci])
    for row in table.rows:
        for ci, cell in enumerate(row.cells):
            cell.width = Cm(widths[ci])

    # A long table split across a page break left its header row stranded alone at
    # the bottom of the previous page while the body rows continued without one.
    # Mark the first row as a repeating header and forbid rows from splitting.
    hdr_trpr = table.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    hdr_trpr.append(tbl_header)
    for row in table.rows:
        trpr = row._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        no_split.set(qn("w:val"), "true")
        trpr.append(no_split)
    for cell in table.rows[0].cells:
        for para in cell.paragraphs:
            para.paragraph_format.keep_with_next = True

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def convert(md_path: Path, docx_path: Path, *, title_for_meta: str, author: str, lang: str = "zh") -> dict:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    doc = Document()
    configure_page(doc)
    configure_styles(doc)

    stats = {"headings": 0, "tables": 0, "code_blocks": 0, "paragraphs": 0, "list_items": 0}

    i = 0
    n = len(lines)
    first_h1 = True
    front_matter = True     # paragraphs before the first real heading
    indent = INDENT_EN if lang == "en" else INDENT_ZH
    body_gap = 0 if lang == "en" else 6
    # English gets left alignment, Chinese gets justification. Justified English
    # without hyphenation stretches inter-word spaces badly on short lines; the
    # reference list was showing gaps several characters wide ("Transactions on
    # Image      Processing"). Chinese has no word spacing to stretch, so
    # justification is the correct choice there.
    body_align = WD_ALIGN_PARAGRAPH.LEFT if lang == "en" else WD_ALIGN_PARAGRAPH.JUSTIFY
    while i < n:
        line = lines[i]

        # fenced code block
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            para = doc.add_paragraph()
            para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            para.paragraph_format.left_indent = Cm(0.6)
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(8)
            shade(para, "F4F5F7")
            text = "\n".join(buf).strip("\n")
            run = para.add_run(text)
            set_run_fonts(run, MONO, CJK_BODY)
            run.font.size = Pt(10)
            stats["code_blocks"] += 1
            continue

        # table
        if line.strip().startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(split_table_row(lines[i]))
                i += 1
            if len(block) >= 2 and is_separator_row(block[1]):
                del block[1]
            add_table(doc, block)
            stats["tables"] += 1
            continue

        stripped = line.strip()

        # horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1 and first_h1:
                p = doc.add_paragraph(style="Title")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_rich_text(p, text, latin=LATIN_HEAD, cjk=CJK_HEAD, base_bold=True)
                for run in p.runs:
                    run.font.size = Pt(20)
                    run.font.color.rgb = RGBColor(0, 0, 0)
                first_h1 = False
            else:
                p = doc.add_paragraph(style=f"Heading {max(1, min(level - 1, 3))}")
                add_rich_text(p, text, latin=LATIN_HEAD, cjk=CJK_HEAD, base_bold=True)
                for run in p.runs:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0, 0, 0)
                front_matter = False
            stats["headings"] += 1
            i += 1
            continue

        # block quote
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            shade(p, "F2F4F8")
            add_rich_text(p, " ".join(x for x in buf if x), latin=LATIN_BODY, cjk=CJK_BODY)
            for run in p.runs:
                run.italic = True
            stats["paragraphs"] += 1
            continue

        # bullet list
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            p.paragraph_format.line_spacing = LINE_SPACING
            p.paragraph_format.space_after = Pt(3)
            add_rich_text(p, m.group(1), latin=LATIN_BODY, cjk=CJK_BODY)
            stats["list_items"] += 1
            i += 1
            continue

        # numbered list
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            # Literal numbering rather than the List Number style. Word shares one
            # numbering sequence across every paragraph using that style, so the
            # fourth list in the document continued at 6, 7, 8 instead of
            # restarting at 1. Writing the markdown number out keeps the source of
            # truth in one place and removes the problem entirely.
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            p.paragraph_format.line_spacing = LINE_SPACING
            p.paragraph_format.left_indent = Cm(0.95)
            p.paragraph_format.first_line_indent = Cm(-0.55)
            p.paragraph_format.space_after = Pt(3)
            num_run = p.add_run(f"{m.group(1)}. ")
            set_run_fonts(num_run, LATIN_BODY, CJK_BODY)
            add_rich_text(p, m.group(2), latin=LATIN_BODY, cjk=CJK_BODY)
            stats["list_items"] += 1
            i += 1
            continue

        # blank
        if not stripped:
            i += 1
            continue

        # ordinary paragraph: join consecutive non-blank, non-structural lines
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", ">", "```"))
                    or re.match(r"^[-*]\s+", nxt) or re.match(r"^\d+\.\s+", nxt)
                    or re.fullmatch(r"-{3,}", nxt)):
                break
            buf.append(nxt)
            i += 1
        p = doc.add_paragraph()
        text = " ".join(buf)
        pf = p.paragraph_format
        if front_matter:
            # Title page block: centred, no indent, on either language.
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.first_line_indent = Cm(0)
            pf.space_after = Pt(6)
        elif re.match(r"^\[\d+\]", text):
            # Reference entry: hanging indent so the number sits in the margin and
            # continuation lines line up under the text, not under the number.
            pf.alignment = body_align
            pf.left_indent = Cm(REF_HANG)
            pf.first_line_indent = Cm(-REF_HANG)
            pf.space_after = Pt(body_gap)
        else:
            pf.alignment = body_align
            pf.first_line_indent = Cm(indent)
            pf.space_after = Pt(body_gap)
        add_rich_text(p, text, latin=LATIN_BODY, cjk=CJK_BODY)
        stats["paragraphs"] += 1

    doc.core_properties.title = title_for_meta
    doc.core_properties.author = author
    doc.core_properties.subject = "Research Proposal"
    doc.save(docx_path)
    return stats


def main() -> int:
    jobs = [
        ("proposal_zh.md", "研究计划书_记忆库压缩与工业报警_吴源泰.docx",
         "记忆库压缩何时改变工业报警——面向边缘部署的决策保持筛选方法", "吴源泰", "zh"),
        ("proposal_en.md", "Research_Proposal_Memory_Compression_and_Industrial_Alarms.docx",
         "When Does Memory Compression Change Industrial Alarms?", "Wu Yuantai", "en"),
    ]
    for src, dst, title, author, lang in jobs:
        stats = convert(HERE / src, HERE / dst, title_for_meta=title, author=author, lang=lang)
        print(f"{dst}")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
