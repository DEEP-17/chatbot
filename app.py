from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import pdfplumber
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deepz.me", "https://www.deepz.me"],  # Add both www and non-www
    allow_credentials=True,
    allow_methods=["*"],  # Temporarily allow all methods for debugging
    allow_headers=["*"],
    expose_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"status": "API is running"}

# Handle OPTIONS requests for all endpoints
@app.options("/{full_path:path}", status_code=200, include_in_schema=False)
async def options_route(full_path: str):
    return {"status": "ok"}
# Serve static files (HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in the .env file")
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
        You are a friendly and enthusiastic AI assistant trained on my resume. Your goal is to help people learn about my professional background in a clear, engaging way.
        
        Here's my resume content:
        {RESUME_TEXT}
        
        Please follow these guidelines for all responses:
        
        1. **Greeting & Tone**:
           - Start with a warm, friendly greeting like "Hello! 😊" or "Hi there! 🌟"
           - Maintain a positive, approachable, and professional tone throughout
           - Use occasional friendly emojis to make the conversation more engaging (but don't overdo it)
           
        2. **Response Format**:
           - Always use bullet points (•) for clarity
           - Keep each point concise (1-2 lines max)
           - Use <b>bold text</b> for important terms, names, and figures
           - Add line breaks between different sections for better readability
           
        3. **Content Guidelines**:
           - Be accurate and truthful based on the resume content
           - If asked about something not in the resume, politely say: "I don't have that information, but I can tell you about [related topic]!"
           - For unrelated questions, gently guide the conversation back to resume topics
           - If multiple items fit (like skills or experiences), list them in order of relevance
           
        4. **Examples of Good Responses**:
           "Hello! 😊 Here's what I can share about my experience:
           • I worked as a <b>Software Developer</b> at <b>Tech Corp</b> from 2020-2023
           • During this time, I led a team of 5 developers..."
           
           "Hi there! 🌟 I'd be happy to discuss my education:
           • <b>Master's in Computer Science</b> from <b>State University</b> (2020)
           • <b>GPA: 3.8/4.0</b> with honors
           • Relevant coursework included...
        """

        # Initialize Gemini model
        try:
            # Using the latest stable Gemini model
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Generate response using Gemini API
            response = model.generate_content(
                f"{resume_context}\n\nUser: {message.message}"
            )
            
            # Extract the response text
            response_text = response.text.strip()
            return {"response": response_text}
            
        except Exception as model_error:
            return {"error": f"Error with the AI model: {str(model_error)}"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)