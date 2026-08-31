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


def make_pdf(lines, sizes=None, page_size=(612, 792)) -> bytes:
    sizes = sizes or [11.0] * len(lines)
    w, h = page_size
    ops = []
    y = h - 72
    for text, size in zip(lines, sizes):
        esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"BT /F1 {size} Tf 72 {y:.1f} Td ({esc}) Tj ET")
        y -= size * 1.6
    stream = "\n".join(ops).encode("latin-1")

    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("latin-1")
    )
    objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

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
