import gradio as gr
import fitz  
import os
import easyocr  
import re  
from huggingface_hub import InferenceClient

HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(
    provider="together",  
    api_key=HF_API_KEY
)

reader = easyocr.Reader(['en'])

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file.name) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text.strip()

def extract_text_from_image(image_file):
    text = reader.readtext(image_file.name, detail=0)
    return "\n".join(text).strip()

def analyze_medical_report(file):
    if file.name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file)
    else:
        text = extract_text_from_image(file)
    
    if not text:
        return "No text found in the uploaded document."

    messages = [
        {"role": "user", "content": f"""
    Analyze the following medical report and provide a structured response in the exact format below. 
    Ensure the response follows this structure, dont use extra space
    **Short Description:**  
    [Briefly summarize the report in 2-3 sentences. Include key test details.]
    **Key Concerns:**
    explain in Ordered form. First tell the test name value and tell is it high or low. Keep it short
    2-3 main concerns are enough
    1. **[Test Name (Value)]:** [High or low Explain what the abnormality suggests.]  

    **Recommendations:**  
    Give in bullet points. Give only 2-3 and dont write too much explanation

    Medical Report:  
    {text}
    """}
    ]

    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1", 
            messages=messages, 
            max_tokens=500,
        )

        output = completion.choices[0].message.content if completion.choices else "No response generated."
        
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
        
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(
    fn=analyze_medical_report,
    inputs=gr.File(type="filepath", label="Upload Medical Report (PDF/Image)"),
    outputs=gr.Textbox(label="AI Analysis"),
    title="AI-Powered Medical Report Analyzer",
    description="Upload your medical report (PDF or Image), and the AI will analyze it to identify potential issues."
)

if __name__ == "__main__":
    interface.launch()

