# 🏥 Medical Report Simplifier using Generative AI

This project simplifies complex medical reports into easy-to-understand summaries using a generative AI model (`facebook/bart-large-cnn`). It features a user-friendly web interface built with Streamlit that allows users to upload PDF medical reports and receive structured, simplified outputs.

---

## 📌 Features

- 📄 Upload and extract medical report PDFs
- 🤖 Summarize and simplify technical medical content using a pre-trained BART model
- 🧾 Structured output with **Introduction**, **Key Findings**, and **Recommendations**
- ⚡ Chunk-wise processing for handling large files efficiently
- 🌐 Lightweight web UI using Streamlit

---

## 🛠️ Tech Stack

- Python 🐍
- [HuggingFace Transformers](https://huggingface.co/) 🤗
- `facebook/bart-large-cnn` pre-trained summarization model
- PyPDF2 (for PDF parsing)
- Streamlit (for the web interface)

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/medical-report-simplifier.git
cd medical-report-simplifier
pip install -r requirements.txt
