import PyPDF2
# For newer Python versions and better handling, consider 'pypdf'
# from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or "" # Handle potential None returns
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into smaller, overlapping chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text): 
            break
    return chunks

if __name__ == '__main__':
    # Example usage:
    # Create a dummy PDF for testing if you don't have one
    # from reportlab.pdfgen import canvas
    # c = canvas.Canvas("dummy.pdf")
    # c.drawString(100, 750, "This is a test document.")
    # c.drawString(100, 730, "It has some content for text extraction.")
    # c.save()

    # extracted_text = extract_text_from_pdf("dummy.pdf")
    # print("Extracted Text:")
    # print(extracted_text)
    #
    # if extracted_text:
    #     chunks = chunk_text(extracted_text)
    #     print("\nText Chunks:")
    #     for i, chunk in enumerate(chunks):
    #         print(f"Chunk {i+1}: {chunk[:100]}...") # Print first 100 chars
