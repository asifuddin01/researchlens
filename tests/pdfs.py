"""A minimal PDF writer, so the upload path can be tested without a corpus.

The corpus PDFs are not committed — they are journal papers and redistributing
them is a licensing question this repository does not need to answer — so a
test that needs a real parse has to make its own. This produces an uncompressed
one-page PDF with a text layer, which is all `parse_bytes` requires.

Not a general PDF library and not trying to be. It exists so that "does an
upload survive parse, chunk, embed and search" is a test rather than a thing
someone checks by hand.
"""

from __future__ import annotations


def make_small_caps_pdf(words, page_size=(612, 792)) -> bytes:
    """A title set the way ICLR and NeurIPS set one.

    `words` is a list of (initial, remainder) pairs. The initial is drawn at
    17.22pt and the remainder at 13.77pt, sharing a baseline — the exact
    geometry measured in the ViT paper, where the two differ by 2.7pt in `top`
    and 0.8pt in `bottom`.
    """
    w, h = page_size
    baseline = h - 100
    # Courier, because every glyph is exactly 0.6em wide and the advance below
    # is therefore exact. With a proportional face the fixture would have to
    # model per-glyph widths, and a wrong model would fail the test for a
    # reason that has nothing to do with the parser.
    ops, x = [], 72.0
    for initial, rest in words:
        ops.append(f"BT /F2 17.22 Tf {x:.2f} {baseline:.2f} Td ({initial}) Tj ET")
        x += 17.22 * 0.6 * len(initial)
        ops.append(f"BT /F2 13.77 Tf {x:.2f} {baseline:.2f} Td ({rest}) Tj ET")
        x += 13.77 * 0.6 * len(rest) + 6.0        # a real inter-word space
    body = "Body text that gives this page a usable density of characters. " * 12
    ops.append(f"BT /F1 11 Tf 72 {baseline - 40:.2f} Td ({body[:250]}) Tj ET")
    return _assemble(ops, page_size)


def make_drop_cap_pdf(page_size=(612, 792)) -> bytes:
    """A modest title, and a body line opening with an outsized initial.

    The shape that regressed when title tiering used a line's maximum size: one
    29.4pt glyph on a 10pt line promoted the whole body line above the title.
    """
    w, h = page_size
    ops = [f"BT /F1 16 Tf 72 {h - 90:.2f} Td (Urolithiasis Classification With Fusion) Tj ET"]
    y = h - 140
    ops.append(f"BT /F2 29.4 Tf 72 {y:.2f} Td (U) Tj ET")
    ops.append(f"BT /F2 10 Tf {72 + 29.4 * 0.6:.2f} {y:.2f} Td "
               "(rolithiasis is introduced to balance the dual objectives) Tj ET")
    body = "Ordinary body text carrying the page to a usable density. " * 10
    for i in range(4):
        ops.append(f"BT /F1 10 Tf 72 {y - 20 * (i + 1):.2f} Td ({body[:110]}) Tj ET")
    return _assemble(ops, page_size)


def make_rotated_margin_pdf(page_size=(612, 792)) -> bytes:
    """A title, plus an arXiv stamp printed sideways up the left margin.

    The stamp's glyphs share an x and spread over many tops, so they scatter
    into whatever lines they land near — which is how seven titles acquired a
    trailing "A V X".
    """
    w, h = page_size
    ops = [f"BT /F1 14.35 Tf 72 {h - 100:.2f} Td "
           "(Swin Transformer Hierarchical Vision Using Shifted Windows) Tj ET"]
    # 90-degree text matrix: the stamp runs up the page at x = 16.
    for i, ch in enumerate("arXiv:2103.14030v2"):
        ops.append(f"BT /F1 14.44 Tf 0 1 -1 0 16 {200 + i * 12:.2f} Tm ({ch}) Tj ET")
    body = "Body text that gives the page a usable character density. " * 10
    for i in range(4):
        ops.append(f"BT /F1 10 Tf 72 {h - 140 - 20 * i:.2f} Td ({body[:110]}) Tj ET")
    return _assemble(ops, page_size)


def make_pdf(lines, sizes=None, page_size=(612, 792)) -> bytes:
    sizes = sizes or [11.0] * len(lines)
    w, h = page_size
    ops = []
    y = h - 72
    for text, size in zip(lines, sizes):
        esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"BT /F1 {size} Tf 72 {y:.1f} Td ({esc}) Tj ET")
        y -= size * 1.6
    return _assemble(ops, page_size)


def _assemble(ops, page_size):
    w, h = page_size
    stream = "\n".join(ops).encode("latin-1")

    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
            "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"
        ).encode("latin-1")
    )
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
