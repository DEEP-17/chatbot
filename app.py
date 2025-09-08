from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import pdfplumber
import os

app = FastAPI()
# Enable CORS to allow requests from HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deepz.me"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static files (HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBoXOTLzLIUNiX0gA7rwEMIB5eAdPaKvuQ")
genai.configure(api_key=GEMINI_API_KEY)
# Extract text from resume PDF or read from resume.txt
def extract_resume_text(pdf_path="resume.pdf", txt_path="resume.txt"):
    try:
        # Check if resume.txt exists
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                return txt_file.read()
        
        # If resume.txt doesn't exist, extract from resume.pdf
        if not os.path.exists(pdf_path):
            return "Error: resume.pdf not found. Please place your resume PDF in the project directory."
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        # Save extracted text to resume.txt
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        
        return text
    
    except FileNotFoundError:
        return "Error: resume.pdf not found. Please place your resume PDF in the project directory."
    except Exception as e:
        return f"Error processing resume: {str(e)}"

RESUME_TEXT = extract_resume_text()

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        # Define the system prompt for resume-based queries
        resume_context = f"""
        You are a chatbot trained on my resume. Answer questions based on the following resume content:
        {RESUME_TEXT}
        Be concise, professional, and accurate. If the question is unrelated to the resume, politely redirect to resume-related topics.
        Format ALL responses as follows:
        - Use bullet points for each distinct piece of information (e.g., each degree, project, skill, job, or achievement).
        - Use bold HTML tags (<b>text</b>) for important figures or words, such as names, dates, institutions, CGPA, percentages, project names, technologies, or key terms.
        - move to next line after every line, and bullet each line.
        - Avoid plain text paragraphs, unformatted lists, or any other format; strictly use bullet points and bold tags for key terms in every response.
        """

        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate response using Gemini API
        response = model.generate_content(
            f"{resume_context}\n\nUser: {message.message}"
        )
        
        # Extract the response text
        response_text = response.text.strip()
        return {"response": response_text}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)