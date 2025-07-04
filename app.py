import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF file.
    """
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None


def simplify_text(text, tokenizer, model):
    """
    Simplify medical text and structure it into a report format.
    """
    max_chunk_length = 1024
    step = 800
    max_summary_length = 700
    min_summary_length = 300

    text_chunks = [
        text[i:i + max_chunk_length]
        for i in range(0, len(text), step)
    ]

    simplified_chunks = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=max_chunk_length, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        simplified_chunks.append(summary)

    # Combine simplified text and format it
    structured_report = structure_report(" ".join(simplified_chunks))
    return structured_report


def structure_report(text):
    """
    Organize the simplified text into a structured report format.
    """
    sections = {
        "Introduction": "This section provides an overview of the report findings in simple language.",
        "Key Findings": "Highlights the main medical observations and details.",
        "Recommendations": "Suggestions and next steps for understanding and managing the condition."
    }

    structured_report = "--- Medical Report ---\n\n"

    # Introduction
    structured_report += f"## {sections['Introduction']}\n\n"
    structured_report += f"{text[:300]}...\n\n"  # First 300 characters as an introduction.

    # Key Findings
    structured_report += f"## {sections['Key Findings']}\n\n"
    findings = text[300:800].split(". ")  # Extract the next part as findings.
    for i, finding in enumerate(findings[:5], start=1):  # Limit to 5 bullet points.
        structured_report += f"- {finding.strip()}.\n"

    # Recommendations
    structured_report += f"\n## {sections['Recommendations']}\n\n"
    recommendations = text[800:1200].split(". ")  # Extract the next part for recommendations.
    for i, recommendation in enumerate(recommendations[:5], start=1):  # Limit to 5 points.
        structured_report += f"{i}. {recommendation.strip()}.\n"

    return structured_report


# Streamlit App
st.title("Medical Report Simplifier")
st.write("Upload a medical report PDF to simplify its content into a more understandable format.")

# Load summarization model
summarization_model_name = "facebook/bart-large-cnn"

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
    return tokenizer, model

tokenizer, model = load_models()

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing the uploaded PDF..."):
        # Extract text from the PDF
        report_text = extract_text_from_pdf(uploaded_file)
        if report_text:
            # Simplify the text
            st.info("Simplifying the text...")
            simplified_report = simplify_text(report_text, tokenizer, model)
            st.success("Simplification complete!")

            # Display the structured report
            st.subheader("Simplified Medical Report:")
            st.text_area("Structured Report:", value=simplified_report, height=500)
