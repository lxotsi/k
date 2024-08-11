import os
import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    extracted_text = ""
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        page_text = page.get_text("text")
        extracted_text += page_text + "\n"
    return extracted_text

def extract_texts_from_multiple_pdfs(directory_path):
    combined_text = ""
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            pdf_text = extract_text_from_pdf(pdf_path)
            combined_text += pdf_text + "\n"
    return combined_text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    # Replace 'path/to/your/pdf/directory' with the actual path to your directory containing PDFs
    pdf_directory_path = "C:/Users/PC/OneDrive/Documents/falcon/datasets/books/biology"
    combined_text_data = extract_texts_from_multiple_pdfs(pdf_directory_path)
    cleaned_combined_text_data = preprocess_text(combined_text_data)

    # Save the cleaned text to a file for later use
    with open("cleaned_text_data_bio.txt", "w", encoding="utf-8") as text_file:
        text_file.write(cleaned_combined_text_data)

