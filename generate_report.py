"""Create the assignment cover/report DOCX and PDF from verified results."""

from __future__ import annotations

import html
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).parent
DOCX = ROOT / "Hisham_Adil_230953434_FGAI_Assignment2.docx"
PDF = ROOT / "Hisham_Adil_230953434_FGAI_Assignment2.pdf"


def paragraph(text: str, style: str = "Normal") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def table(rows: list[list[str]]) -> str:
    result = ["<w:tbl>"]
    for row in rows:
        result.append("<w:tr>")
        for cell in row:
            result.append(f"<w:tc><w:p><w:r><w:t>{escape(cell)}</w:t></w:r></w:p></w:tc>")
        result.append("</w:tr>")
    result.append("</w:tbl>")
    return "".join(result)


def create_docx() -> None:
    body = [
        paragraph("Foundations of Generative AI (ICT4403)", "Title"),
        paragraph("Assignment 2: Autoregressive Text Generation", "Heading1"),
        paragraph("Student name: Hisham Adil"),
        paragraph("Registration number: 230953434"),
        paragraph("System title: Comparative Word-Level Autoregressive Text Generation using Simple RNN, Bidirectional RNN, LSTM and GRU"),
        paragraph("Dataset: Alice's Adventures in Wonderland by Lewis Carroll"),
        paragraph("Dataset source: https://www.gutenberg.org/files/11/11-0.txt"),
        paragraph("GitHub repository: https://github.com/Hisham-byte/RNNTextGeneration"),
        paragraph("Project summary", "Heading1"),
        paragraph("The system tokenizes the literary text, constructs overlapping windows of 20 input words and one next-word target, and trains four models under comparable conditions. It evaluates next-word loss, perplexity, accuracy and training time, and generates text from two fixed seed sequences at temperatures 0.5, 1.0 and 1.5."),
        paragraph("Verified results", "Heading1"),
        table([
            ["Model", "Test loss", "Perplexity", "Accuracy", "Time (s)"],
            ["GRU", "4.7548", "116.1440", "18.92%", "74.32"],
            ["SimpleRNN", "4.8733", "130.7514", "18.20%", "31.07"],
            ["LSTM", "4.9211", "137.1574", "17.91%", "101.61"],
            ["Bidirectional RNN", "5.0459", "155.3812", "16.66%", "42.69"],
        ]),
        paragraph("Conclusion", "Heading1"),
        paragraph("The GRU is the best overall model for this run because it achieved the lowest test loss and perplexity and the highest next-word accuracy. Temperature 0.5 produced the most coherent but repetitive text, temperature 1.0 provided the best balance of coherence and diversity, and temperature 1.5 increased diversity while reducing grammatical consistency."),
        paragraph("Execution instructions", "Heading1"),
        paragraph("Install the dependencies from requirements.txt and run: python train_text_generation.py --epochs 20 --output-dir results. The repository contains the implementation, metrics, generated samples, dataset summary and validation-loss plot."),
    ]
    document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + "".join(body) + '<w:sectPr/></w:body></w:document>'
    styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style></w:styles>'
    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    relationships = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    document_relationships = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/_rels/document.xml.rels", document_relationships)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)


def create_pdf() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    story = [Paragraph("Foundations of Generative AI (ICT4403)", styles["Title"]), Paragraph("Assignment 2: Autoregressive Text Generation", styles["Heading1"])]
    story += [Paragraph(text, styles["BodyText"]) for text in [
        "Student name: Hisham Adil",
        "Registration number: 230953434",
        "System title: Comparative Word-Level Autoregressive Text Generation using Simple RNN, Bidirectional RNN, LSTM and GRU",
        "Dataset: Alice's Adventures in Wonderland by Lewis Carroll",
        "Dataset source: https://www.gutenberg.org/files/11/11-0.txt",
        "GitHub repository: https://github.com/Hisham-byte/RNNTextGeneration",
    ]]
    story += [Spacer(1, 12), Paragraph("Verified results", styles["Heading1"])]
    data = [["Model", "Loss", "Perplexity", "Accuracy", "Time (s)"], ["GRU", "4.7548", "116.1440", "18.92%", "74.32"], ["SimpleRNN", "4.8733", "130.7514", "18.20%", "31.07"], ["LSTM", "4.9211", "137.1574", "17.91%", "101.61"], ["Bidirectional RNN", "5.0459", "155.3812", "16.66%", "42.69"]]
    result_table = Table(data)
    result_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story += [result_table, Spacer(1, 12), Paragraph("Conclusion", styles["Heading1"]), Paragraph("GRU is the best overall model because it achieved the lowest loss and perplexity and the highest accuracy. Temperature 0.5 was most coherent but repetitive, 1.0 balanced coherence and diversity, and 1.5 increased diversity with reduced grammatical consistency.", styles["BodyText"])]
    SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45).build(story)


if __name__ == "__main__":
    create_docx()
    create_pdf()
    print(DOCX)
    print(PDF)
