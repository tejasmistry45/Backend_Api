import os
from langchain_together import Together
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
from .logger_function import logger_function


filename=os.path.basename(__file__)[:-3]

load_dotenv()
# print("API Key:", os.getenv("TOGATHER_API_KEY"))

# Initialize the Together AI LLM
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    # model="meta-llama/Llama-2-70b-hf",
    # model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    # model= "mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0,
    max_tokens=512,
    together_api_key=os.getenv("TOGATHER_API_KEY"),
)

template = """
You are an expert resume analyzer trained to extract structured information from unstructured resume text.

Analyze the resume text below and extract the following details as accurately as possible:

1. **Full Name** of the candidate (based on patterns like headers, email signatures, etc.)
2. **Total Years of Professional Experience** (You may estimate based on work history, graduation year, and job descriptions if not explicitly mentioned.)
3. **Key Skills** (Extract relevant job-related skills such as Project Management, Data Analysis, Cloud Computing, etc. — comma separated)
4. **Technologies and Tools** (List technologies, frameworks, programming languages, libraries, cloud platforms, software, or tools mentioned — comma separated)
5. **Estimated Experience Level** — Choose **Junior** (0–2 years), **Mid** (2–5 years), or **Senior** (5+ years). Make your best estimate based on job history and roles/responsibilities.

Example output:
Name: John Doe  
Years of Experience: 4  
Key Skills: Data Analysis, Machine Learning, Project Management  
Technologies / Tools: Python, SQL, TensorFlow, Excel, Git  
Estimated Experience Level: Mid

Now extract the same details from the following resume text:

---
{resume_text}
---

Please provide the results in the following format:

Name:  
Years of Experience:  
Key Skills:  
Technologies / Tools:  
Estimated Experience Level:  
"""


prompt = PromptTemplate(
    input_variables=["resume_text"],
    template=template,
)

# Adjusting the response parsing
def extract_resume_info(resume_text):
    # Format the resume text to fit in the prompt
    final_prompt = prompt.format(resume_text=resume_text[:700])  # limit the input size
    
    # Get the response from the LLM
    raw_response = llm.invoke(final_prompt)
    
    # Try to parse the raw response into a structured format
    try:
        response = json.loads(raw_response)  # if response is valid JSON
        name = response.get("Name", "Not Available")
        years = response.get("Years of Experience", "Not Available")
        skills = response.get("Key Skills", "Not Available")
        tech = response.get("Technologies / Tools", "Not Available")
        level = response.get("Estimated Experience Level", "Not Available")
    except json.JSONDecodeError:
        # Fallback for plain text
        response_lines = raw_response.strip().split("\n")
        data = {}
        for line in response_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        
        name = data.get("Name", "Not Available")
        years = data.get("Years of Experience", "Not Available")
        skills = data.get("Key Skills", "Not Available")
        tech = data.get("Technologies / Tools", "Not Available")
        level = data.get("Estimated Experience Level", "Not Available")

        # Return as list of key-value dicts
        summary = [
            {"label": "Name", "value": name},
            {"label": "Years of Experience", "value": years},
            {"label": "Key Skills", "value": skills},
            {"label": "Technologies / Tools", "value": tech},
            {"label": "Estimated Experience Level", "value": level},
        ]

        return summary