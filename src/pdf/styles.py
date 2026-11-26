"""Shared CSS stylesheets for exam, Korean/English analysis, and vocabulary PDFs."""

from __future__ import annotations

EXAM_CSS = """
body {
    font-family: "MogiKorSans", "NanumGothic", "Malgun Gothic", sans-serif;
    font-size: 9.4pt;
    line-height: 1.52;
    color: #0F172A;
    margin: 0;
    padding: 0;
}
sub {
    font-size: 72%;
    vertical-align: sub;
}
sup {
    font-size: 72%;
    vertical-align: super;
}
.exam-banner {
    border: 1.2pt solid #1E293B;
    border-radius: 4pt;
    padding: 7pt 10pt;
    background-color: #F8FAFC;
}
.exam-banner-sub {
    font-size: 8.2pt;
    font-weight: bold;
    color: #1D4ED8;
    letter-spacing: 0.04em;
}
.exam-banner-title {
    font-size: 14.5pt;
    font-weight: bold;
    color: #0F172A;
    margin-top: 2pt;
}
.student-box {
    width: 100%;
    border-collapse: collapse;
    margin-top: 5pt;
    font-size: 8.4pt;
    background-color: #FFFFFF;
}
.student-box td {
    border: 0.7pt solid #475569;
    padding: 3pt 6pt;
    color: #1E293B;
}
.double-rule-top {
    border-top: 2pt solid #0F172A;
    margin-top: 6pt;
}
.double-rule-bot {
    border-top: 0.7pt solid #334155;
    margin-top: 2pt;
}
.q-wrap {
    margin-bottom: 6pt;
}
.q-header {
    margin-bottom: 4pt;
    font-size: 9.2pt;
    color: #0F172A;
}
.q-num {
    font-weight: bold;
    font-size: 11.2pt;
    color: #1E3A8A;
}
.badge {
    background-color: #DBEAFE;
    color: #1E40AF;
    font-size: 7.4pt;
    padding: 1pt 4.5pt;
    border-radius: 3pt;
    font-weight: bold;
}
.badge-orig {
    background-color: #DCFCE7;
    color: #166534;
    font-size: 7.4pt;
    padding: 1pt 4.5pt;
    border-radius: 3pt;
    font-weight: bold;
}
.meta {
    color: #475569;
    font-size: 7.8pt;
}
.stem-p {
    margin-top: 2pt;
    margin-bottom: 4pt;
    word-break: keep-all;
}
.stem-line {
    margin-bottom: 2.5pt;
}
.stem-line:last-child { margin-bottom: 0; }
.data-line {
    margin-bottom: 2.2pt;
}
.data-line:last-child { margin-bottom: 0; }
.pdf-data-box {
    border: 0.8pt solid #94A3B8;
    border-left: 2.6pt solid #334155;
    background-color: #F8FAFC;
    padding: 5pt 7pt;
    margin: 5pt 0;
    border-radius: 3pt;
    font-size: 8.9pt;
    line-height: 1.48;
}
.pdf-data-title {
    font-size: 7.8pt;
    font-weight: bold;
    color: #334155;
    margin-bottom: 2pt;
}
.pdf-bogi-box {
    border: 1pt solid #1E293B;
    padding: 5pt 7pt;
    margin: 5pt 0;
    background-color: #FFFFFF;
}
.pdf-bogi-title {
    text-align: center;
    font-weight: bold;
    font-size: 8.6pt;
    letter-spacing: 0.25em;
    color: #0F172A;
    border-bottom: 0.6pt solid #CBD5E1;
    padding-bottom: 2.5pt;
    margin-bottom: 4pt;
}
.bogi-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    line-height: 1.48;
}
.bogi-tbl td {
    vertical-align: top;
    padding: 1.2pt 0;
}
.bogi-mk {
    width: 16pt;
    font-weight: bold;
    color: #1E3A8A;
}
.choices {
    margin-top: 4pt;
    margin-left: 4pt;
}
.choice-item {
    margin-bottom: 2.5pt;
    word-break: keep-all;
}
.choice-grid {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    font-size: 9.1pt;
}
.choice-grid td {
    padding: 2pt 4pt 2pt 0;
    vertical-align: top;
    word-break: keep-all;
}
.ref-note {
    margin-top: 4pt;
    color: #1E40AF;
    font-weight: bold;
    font-size: 7.6pt;
}
.vis-note {
    margin-top: 2pt;
    padding: 3pt 5pt;
    background-color: #FFFBEB;
    border-left: 2pt solid #D97706;
    color: #92400E;
    font-size: 7.6pt;
}
.ans-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    margin-bottom: 10pt;
    font-size: 8.5pt;
}
.ans-table th, .ans-table td {
    border: 0.65pt solid #94A3B8;
    padding: 3.5pt 5pt;
    text-align: center;
}
.ans-table th {
    background-color: #EFF6FF;
    color: #1E3A8A;
    font-weight: bold;
}
.exp-card {
    border: 0.7pt solid #CBD5E1;
    background-color: #F8FAFC;
    padding: 6pt 7pt;
    margin-bottom: 6pt;
    border-radius: 4pt;
    font-size: 8.7pt;
    line-height: 1.48;
}
.exp-title {
    font-weight: bold;
    font-size: 9.2pt;
    color: #0F172A;
    margin-bottom: 3pt;
    border-bottom: 0.5pt solid #E2E8F0;
    padding-bottom: 2pt;
}
.exp-ans {
    color: #DC2626;
    font-weight: bold;
}
.exp-sub {
    font-size: 8pt;
    color: #1D4ED8;
    margin-bottom: 3pt;
}
.eng-u {
    text-decoration: underline;
    font-weight: bold;
}
.eng-blank {
    text-decoration: underline;
    font-weight: bold;
}
.eng-inst {
    font-weight: bold;
    margin-top: 2pt;
    margin-bottom: 4pt;
    word-break: keep-all;
}
.eng-given-box {
    border: 1pt solid #1E293B;
    background-color: #F8FAFC;
    padding: 5pt 7pt;
    margin: 4pt 0 5pt 0;
    font-size: 9.1pt;
    line-height: 1.5;
}
.eng-given-title {
    font-size: 7.6pt;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 2pt;
}
.eng-passage {
    margin-top: 2pt;
    margin-bottom: 4pt;
    font-size: 9.2pt;
    line-height: 1.52;
}
.eng-order-tbl {
    width: 100%;
    border-collapse: collapse;
    margin: 4pt 0;
    font-size: 9.1pt;
    line-height: 1.5;
}
.eng-order-tbl td {
    vertical-align: top;
    padding: 2.5pt 0;
}
.eng-order-mk {
    width: 20pt;
    font-weight: bold;
    color: #1E3A8A;
}
.eng-summary-arrow {
    text-align: center;
    font-weight: bold;
    font-size: 9.5pt;
    color: #334155;
    margin: 2pt 0;
}
.eng-summary-box {
    border: 1pt solid #1E3A8A;
    background-color: #F8FAFC;
    padding: 5pt 7pt;
    margin: 2pt 0 5pt 0;
    font-size: 9.1pt;
    line-height: 1.5;
}
.eng-footnote {
    font-size: 7.8pt;
    color: #475569;
    margin-top: 2pt;
    margin-bottom: 3pt;
}
.bogi-prose {
    font-size: 9pt;
    line-height: 1.48;
    color: #0F172A;
    word-break: keep-all;
}
.bogi-prose-line {
    margin-bottom: 2.2pt;
}
.bogi-prose-line:last-child { margin-bottom: 0; }
.kor-u {
    text-decoration: underline;
    font-weight: bold;
    color: #0F172A;
}
.kor-mk {
    font-weight: bold;
    color: #1E3A8A;
}
.kor-set-intro {
    font-size: 9.4pt;
    font-weight: bold;
    color: #0F172A;
    margin-bottom: 4pt;
    word-break: keep-all;
}
.kor-passage-subhead {
    font-weight: bold;
    font-size: 9.2pt;
    color: #0F172A;
    margin: 4pt 0 2pt 0;
}
.kor-passage-p {
    font-size: 9.1pt;
    line-height: 1.56;
    text-indent: 9pt;
    text-align: justify;
    margin: 0 0 3.5pt 0;
    color: #0F172A;
    word-break: keep-all;
}
.kor-passage-p-cont {
    font-size: 9.1pt;
    line-height: 1.56;
    text-indent: 0;
    text-align: justify;
    margin: 0 0 3.5pt 0;
    color: #0F172A;
    word-break: keep-all;
}
.kor-passage-lit {
    font-size: 9.1pt;
    line-height: 1.56;
    margin: 0 0 1.5pt 0;
    color: #0F172A;
    word-break: keep-all;
}
"""

_KOREAN_ANALYSIS_CSS = """
    body {
        font-family: "NanumGothic", "Malgun Gothic", sans-serif;
        font-size: 9.4pt;
        line-height: 1.54;
        color: #0F172A;
    }
    .hdr-banner {
        border: 1.4pt solid #1E293B;
        border-radius: 5pt;
        background-color: #F8FAFC;
        padding: 9pt 12pt;
        margin-bottom: 10pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .hdr-sub {
        font-size: 8.2pt;
        font-weight: bold;
        color: #1D4ED8;
    }
    .hdr-title {
        font-size: 14.5pt;
        font-weight: bold;
        color: #0F172A;
        margin-top: 2pt;
    }
    .hdr-meta {
        font-size: 8.2pt;
        color: #475569;
        margin-top: 3pt;
    }
    .sec-title {
        font-size: 10.4pt;
        font-weight: bold;
        color: #1E3A8A;
        border-left: 3.5pt solid #2563EB;
        padding-left: 6pt;
        margin-top: 10pt;
        margin-bottom: 5pt;
        break-after: avoid;
        page-break-after: avoid;
    }
    .sec-keep-tbl {
        width: 100%;
        border-collapse: collapse;
        border: none;
        margin: 0;
        padding: 0;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .sec-keep-tr {
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .sec-keep-td {
        padding: 0;
        margin: 0;
        border: none;
        vertical-align: top;
    }
    .theme-box {
        background-color: #EFF6FF;
        border: 0.9pt solid #93C5FD;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-bottom: 6pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .flow-box {
        background-color: #F0FDF4;
        border: 0.9pt solid #86EFAC;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-bottom: 8pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .flow-lines {
        margin-top: 2pt;
        line-height: 1.58;
    }
    .para-card {
        border-bottom: 0.6pt solid #E2E8F0;
        padding: 4pt 0;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .para-lbl {
        font-weight: bold;
        color: #1E40AF;
    }
    .para-kp {
        color: #0F766E;
        font-size: 8.7pt;
        margin-top: 1.5pt;
    }
    .vocab-list {
        border: 0.8pt solid #CBD5E1;
        border-radius: 4pt;
        background-color: #FFFFFF;
        padding: 4pt 8pt;
        margin-bottom: 8pt;
    }
    .vocab-row {
        border-bottom: 0.5pt solid #E2E8F0;
        padding: 4pt 0;
        font-size: 9.0pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .vocab-row-last {
        padding: 4pt 0;
        font-size: 9.0pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .vocab-badge {
        display: inline;
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: bold;
        padding: 1pt 5pt;
        border-radius: 3pt;
    }
    .checkpoint-box {
        background-color: #FFFBEB;
        border: 0.9pt solid #FCD34D;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-bottom: 8pt;
    }
    .note-li {
        margin-bottom: 3pt;
    }
    .passage-box {
        background-color: #FAFAFA;
        border: 0.8pt solid #CBD5E1;
        border-radius: 4pt;
        padding: 8pt 11pt;
        font-size: 9.0pt;
        line-height: 1.62;
    }
    .passage-p {
        margin: 0 0 7.5pt 0;
        text-indent: 9pt;
        text-align: justify;
    }
    .passage-lit-line {
        margin: 0 0 2.5pt 0;
    }
    .passage-hdr {
        margin: 4pt 0 3pt 0;
        font-weight: bold;
        color: #1E293B;
    }
    .page-break {
        page-break-before: always;
    }
    .diagram-wrap {
        margin-top: 4pt;
        margin-bottom: 8pt;
        text-align: center;
    }
    .diagram-img {
        width: 505pt;
        max-width: 100%;
        height: auto;
    }
    .poem-ul {
        font-weight: bold;
        color: #1E3A8A;
        text-decoration: underline;
    }
    .poem-badge {
        font-size: 7.5pt;
        font-weight: bold;
        color: #1D4ED8;
    }
    .poem-notes-box {
        background-color: #EFF6FF;
        border: 0.8pt solid #BFDBFE;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-top: 6pt;
        margin-bottom: 8pt;
    }
    .poem-notes-hdr {
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 4pt;
        font-size: 9.2pt;
    }
    .poem-note-row {
        margin-bottom: 3.5pt;
        font-size: 8.9pt;
        line-height: 1.5;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .eng-sent-card {
        border: 0.8pt solid #CBD5E1;
        border-radius: 4pt;
        background-color: #FFFFFF;
        padding: 6.5pt 9.5pt;
        margin-bottom: 6pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .eng-sent-card-topic {
        border: 1.2pt solid #F59E0B;
        border-left: 3.5pt solid #D97706;
        border-radius: 4pt;
        background-color: #FFFBEB;
        padding: 6.5pt 9.5pt;
        margin-bottom: 6pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .eng-sent-hdr {
        margin-bottom: 2.5pt;
    }
    .eng-sent-num {
        font-weight: bold;
        color: #1E3A8A;
        font-size: 9.5pt;
    }
    .eng-topic-badge {
        display: inline;
        background-color: #FEF3C7;
        color: #B45309;
        border: 0.6pt solid #F59E0B;
        font-weight: bold;
        font-size: 7.8pt;
        padding: 1pt 4.5pt;
        border-radius: 3pt;
    }
    .eng-sent-chunk {
        font-size: 9.4pt;
        font-weight: bold;
        color: #0F172A;
        line-height: 1.56;
        margin-bottom: 2.5pt;
    }
    .eng-slash {
        color: #DC2626;
        font-weight: bold;
    }
    .eng-bracket {
        color: #1D4ED8;
        font-weight: bold;
    }
    .eng-paren {
        color: #0F766E;
        font-weight: bold;
    }
    .eng-sent-kor {
        font-size: 8.9pt;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 3pt;
    }
    .eng-syntax-box {
        background-color: #F8FAFC;
        border-top: 0.5pt solid #E2E8F0;
        padding: 3.5pt 5pt 1pt 5pt;
        margin-top: 2pt;
    }
    .eng-syntax-li {
        font-size: 8.4pt;
        color: #1E40AF;
        line-height: 1.44;
        margin-bottom: 1.5pt;
    }
    """

_VOCAB_PDF_CSS = """
body {
    font-family: "NanumGothic", "Malgun Gothic", "DejaVu Sans", sans-serif;
    font-size: 9pt;
    line-height: 1.45;
    color: #0F172A;
}
.v-banner {
    border: 1.3pt solid #1E293B;
    border-radius: 5pt;
    background-color: #F8FAFC;
    padding: 8pt 11pt;
    margin-bottom: 10pt;
}
.v-sub {
    font-size: 8pt;
    font-weight: bold;
    color: #1D4ED8;
}
.v-title {
    font-size: 14pt;
    font-weight: bold;
    color: #0F172A;
    margin-top: 2pt;
}
.v-meta-tbl {
    width: 100%;
    border-collapse: collapse;
    margin-top: 5pt;
    font-size: 8.2pt;
    background-color: #FFFFFF;
}
.v-meta-tbl td {
    border: 0.6pt solid #64748B;
    padding: 2.5pt 6pt;
    color: #334155;
}
.study-tbl {
    width: 100%;
    border-collapse: separate;
    border-spacing: 5pt 5pt;
}
.study-cell {
    width: 50%;
    vertical-align: top;
    border: 0.75pt solid #CBD5E1;
    border-radius: 4pt;
    background-color: #FFFFFF;
    padding: 5.5pt 7pt;
}
.study-empty {
    width: 50%;
}
.w-num {
    font-size: 8.2pt;
    font-weight: bold;
    color: #2563EB;
}
.w-head {
    font-size: 10.8pt;
    font-weight: bold;
    color: #0F172A;
}
.w-ipa {
    font-size: 8pt;
    color: #64748B;
}
.w-pos {
    font-size: 7.4pt;
    font-weight: bold;
    color: #1E40AF;
    background-color: #DBEAFE;
    padding: 0.5pt 4pt;
    border-radius: 3pt;
}
.w-lvl {
    font-size: 7.2pt;
    font-weight: bold;
    color: #166534;
    background-color: #DCFCE7;
    padding: 0.5pt 4pt;
    border-radius: 3pt;
}
.w-mean {
    font-size: 9pt;
    font-weight: bold;
    color: #1E293B;
    margin-top: 2.5pt;
}
.w-syn {
    font-size: 7.7pt;
    color: #475569;
    margin-top: 2pt;
}
.w-ex {
    font-size: 7.6pt;
    color: #334155;
    margin-top: 2.5pt;
    padding-top: 2pt;
    border-top: 0.5pt dashed #E2E8F0;
}
.test-tbl {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    font-size: 8.8pt;
}
.test-tbl th {
    background-color: #1E3A8A;
    color: #FFFFFF;
    font-weight: bold;
    text-align: left;
    padding: 4.5pt 6pt;
    border: 0.6pt solid #1E3A8A;
    font-size: 8.4pt;
}
.test-tbl td {
    padding: 6.5pt 6pt;
    border: 0.6pt solid #CBD5E1;
    vertical-align: middle;
}
.test-alt td {
    background-color: #F8FAFC;
}
.blank-line {
    color: #64748B;
    font-size: 8.5pt;
}
.ans-break {
    page-break-before: always;
}
.ans-hdr {
    border-bottom: 1.5pt solid #1E3A8A;
    padding-bottom: 4pt;
    margin-bottom: 8pt;
    font-size: 12pt;
    font-weight: bold;
    color: #1E3A8A;
}
.ans-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.2pt;
}
.ans-tbl th {
    background-color: #EFF6FF;
    color: #1E3A8A;
    border: 0.6pt solid #93C5FD;
    padding: 3.5pt 5pt;
    text-align: left;
}
.ans-tbl td {
    border: 0.6pt solid #CBD5E1;
    padding: 3.5pt 5pt;
    vertical-align: top;
}
"""
