# app.py - Flask API for Quiz Solver (Fixed version)
from flask import Flask, request, jsonify
import requests
from playwright.sync_api import sync_playwright
import base64
import json
import os
import time
from datetime import datetime
import PyPDF2
import pandas as pd
import io
import re
from groq import Groq

app = Flask(__name__)

# Configuration
YOUR_EMAIL = os.environ.get('EMAIL', 'your-email@example.com')
YOUR_SECRET = os.environ.get('SECRET', 'your-secret-string')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'your-groq-api-key')

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"


def scrape_quiz_page(url):
    """Scrape and render JavaScript-based quiz page using Playwright"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)  # Wait for JS to execute
            
            # Try to get content from result div
            try:
                quiz_content = page.locator("#result").inner_text()
            except:
                quiz_content = page.locator("body").inner_text()
            
            return quiz_content
        finally:
            browser.close()


def call_groq_api(prompt, system_prompt=None):
    """Call Groq API for analysis"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")


def download_file(url):
    """Download file from URL"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def extract_pdf_text(pdf_content, page_number=None):
    """Extract text from PDF"""
    pdf_file = io.BytesIO(pdf_content)
    reader = PyPDF2.PdfReader(pdf_file)
    
    if page_number:
        return reader.pages[page_number - 1].extract_text()
    else:
        return '\n'.join([page.extract_text() for page in reader.pages])


def extract_pdf_tables(pdf_content):
    """Extract tables from PDF using PyPDF2"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        all_tables = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            lines = text.split('\n')
            if len(lines) > 2:
                all_tables.append({
                    'page': page_num + 1,
                    'text': text,
                    'lines': lines
                })
        
        return all_tables
    except:
        return []


def solve_quiz(quiz_content, quiz_url):
    """Use Groq to solve the quiz"""
    system_prompt = """You are a data analysis expert solving quiz questions.

CRITICAL INSTRUCTIONS:
1. Read the question carefully to understand what VALUE is being asked for
2. The "answer" field in the JSON payload is what YOU need to calculate/provide
3. DO NOT return the entire JSON structure shown in examples - only return the ANSWER VALUE
4. Look for patterns like:
   - "What is the sum of..." → return a number
   - "Download file and analyze..." → return the computed result
   - "answer": <some_value> → YOU need to figure out what <some_value> should be

Return your response as JSON:
{
    "reasoning": "step by step explanation",
    "answer": <the actual answer value - number, string, boolean, or object>,
    "files_to_download": ["url1", "url2"],
    "answer_type": "number|string|boolean|object",
    "page_number": <page number if specified, or null>
}

EXAMPLES:
- Question: "What is 2+2? Submit answer as number" → answer: 4
- Question: "What is the sum of column X?" → answer: 12345 (the calculated sum)
- Question: "Download file, count rows on page 2" → answer: 42 (the count)
"""

    prompt = f"""Quiz Content:
{quiz_content}

Quiz URL: {quiz_url}

Analyze this quiz and determine:
1. What is the actual question being asked?
2. What type of answer is expected (number, string, boolean, object)?
3. Are there any files to download?
4. What page number should be analyzed (if applicable)?

IMPORTANT: The "answer" field in example JSON payloads is a PLACEHOLDER. 
You need to figure out what the actual answer should be based on the question."""

    response = call_groq_api(prompt, system_prompt)
    
    # Try to parse JSON response
    try:
        json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        else:
            return json.loads(response)
    except Exception as e:
        print(f"JSON parse error: {e}, raw response: {response}")
        return {
            "reasoning": response, 
            "answer": response, 
            "files_to_download": [],
            "answer_type": "string"
        }


def process_data_analysis(quiz_content, data, data_type="text"):
    """Analyze data using Groq"""
    prompt = f"""Quiz Question:
{quiz_content}

Data Type: {data_type}
Data:
{str(data)[:5000]}

Based on the question, calculate and provide the EXACT answer.
Examples:
- "What is the sum of the 'value' column?" → Calculate sum and return ONLY the number
- "How many rows are there?" → Count rows and return ONLY the count
- "What is the average?" → Calculate average and return ONLY the number

Return as JSON: {{"answer": <value>, "reasoning": "<explanation>"}}

CRITICAL: Return ONLY the computed value, not descriptions or explanations in the answer field."""

    response = call_groq_api(prompt)
    
    try:
        json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            result = json.loads(response)
        return result.get('answer', response)
    except:
        # Try to extract number
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            num_str = numbers[0]
            return float(num_str) if '.' in num_str else int(num_str)
        return response


def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit answer to the quiz endpoint"""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    print(f"Submitting to {submit_url}:")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(submit_url, json=payload, timeout=30)
    return response.json()


def solve_quiz_chain(initial_url, email, secret):
    """Solve a chain of quizzes"""
    current_url = initial_url
    results = []
    start_time = datetime.now()
    max_duration = 170  # 2:50 to leave buffer
    
    while current_url:
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > max_duration:
            print(f"Time limit reached: {elapsed}s")
            break
        
        print(f"\n{'='*60}")
        print(f"Solving quiz at: {current_url}")
        print(f"Elapsed time: {elapsed:.1f}s")
        print(f"{'='*60}")
        
        try:
            # Step 1: Scrape quiz page
            print("Step 1: Scraping quiz page...")
            quiz_content = scrape_quiz_page(current_url)
            print(f"Quiz content length: {len(quiz_content)}")
            print(f"Quiz content:\n{quiz_content}\n")
            
            # Step 2: Parse to find submit URL
            submit_url_match = re.search(r'(?:POST|Post|post).*?(?:to|TO)\s+(https?://[^\s\)]+)', quiz_content, re.IGNORECASE)
            if submit_url_match:
                submit_url = submit_url_match.group(1).rstrip(',.')
            else:
                submit_url_match = re.search(r'(https?://[^\s]+/submit[^\s]*)', quiz_content)
                submit_url = submit_url_match.group(1).rstrip(',.') if submit_url_match else None
            
            print(f"Submit URL: {submit_url}")
            
            # Step 3: Solve quiz with LLM
            print("\nStep 2: Solving with LLM...")
            solution = solve_quiz(quiz_content, current_url)
            print(f"Initial solution: {json.dumps(solution, indent=2)}")
            
            # Step 4: Handle file downloads if needed
            if solution.get('files_to_download'):
                for file_url in solution['files_to_download']:
                    print(f"\nStep 3: Downloading file from {file_url}...")
                    
                    try:
                        file_content = download_file(file_url)
                        print(f"Downloaded {len(file_content)} bytes")
                        
                        # Process based on file type
                        if file_url.endswith('.pdf'):
                            page_num = solution.get('page_number')
                            
                            if page_num:
                                print(f"Extracting text from page {page_num}...")
                                text = extract_pdf_text(file_content, page_num)
                            else:
                                print("Extracting text from all pages...")
                                text = extract_pdf_text(file_content)
                            
                            tables = extract_pdf_tables(file_content)
                            print(f"Found {len(tables)} potential tables")
                            
                            print("Re-analyzing with PDF content...")
                            answer = process_data_analysis(
                                quiz_content, 
                                {"text": text[:3000], "tables": tables},
                                "pdf"
                            )
                            solution['answer'] = answer
                            
                        elif file_url.endswith('.csv'):
                            print("Processing CSV file...")
                            df = pd.read_csv(io.BytesIO(file_content))
                            print(f"CSV shape: {df.shape}")
                            print(f"Columns: {df.columns.tolist()}")
                            print(f"First few rows:\n{df.head()}")
                            
                            answer = process_data_analysis(
                                quiz_content, 
                                f"Columns: {df.columns.tolist()}\n{df.to_string()[:3000]}",
                                "csv"
                            )
                            solution['answer'] = answer
                            
                        elif file_url.endswith(('.xlsx', '.xls')):
                            print("Processing Excel file...")
                            df = pd.read_excel(io.BytesIO(file_content))
                            print(f"Excel shape: {df.shape}")
                            print(f"Columns: {df.columns.tolist()}")
                            
                            answer = process_data_analysis(
                                quiz_content,
                                f"Columns: {df.columns.tolist()}\n{df.to_string()[:3000]}",
                                "excel"
                            )
                            solution['answer'] = answer
                            
                    except Exception as e:
                        print(f"Error processing file: {e}")
                        import traceback
                        traceback.print_exc()
            
            print(f"\nFinal answer: {solution['answer']} (type: {type(solution['answer']).__name__})")
            
            # Step 5: Submit answer
            if submit_url:
                print(f"\nStep 4: Submitting answer...")
                result = submit_answer(submit_url, email, secret, current_url, solution['answer'])
                print(f"Result: {json.dumps(result, indent=2)}")
                
                results.append({
                    'url': current_url,
                    'answer': solution['answer'],
                    'correct': result.get('correct'),
                    'reason': result.get('reason')
                })
                
                if result.get('correct'):
                    print("✅ Answer CORRECT!")
                else:
                    print(f"❌ Answer WRONG: {result.get('reason')}")
                
                current_url = result.get('url')
                if not current_url:
                    print("No more URLs - quiz complete!")
                    break
            else:
                print("No submit URL found!")
                break
                
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'url': current_url,
                'error': str(e)
            })
            break
    
    return results


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY and GROQ_API_KEY != 'your-groq-api-key'),
        "playwright_available": True
    }), 200


@app.route('/quiz', methods=['POST'])
def handle_quiz():
    """Main endpoint to handle quiz requests"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        if data.get('secret') != YOUR_SECRET:
            return jsonify({"error": "Invalid secret"}), 403
        
        email = data.get('email', YOUR_EMAIL)
        quiz_url = data.get('url')
        
        if not quiz_url:
            return jsonify({"error": "Missing quiz URL"}), 400
        
        print(f"\n{'='*60}")
        print(f"Received quiz request")
        print(f"Email: {email}")
        print(f"Quiz URL: {quiz_url}")
        print(f"{'='*60}")
        
        results = solve_quiz_chain(quiz_url, email, YOUR_SECRET)
        
        return jsonify({
            "status": "success",
            "results": results
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════╗
║     TDS QUIZ SOLVER                                        ║
╚════════════════════════════════════════════════════════════╝

Running on: http://localhost:5001

Endpoints:
- GET  /health - Health check
- POST /quiz   - Main quiz endpoint

Configuration:
- EMAIL: {email}
- SECRET: {secret_set}
- GROQ_API_KEY: {groq_set}
""".format(
    email=YOUR_EMAIL,
    secret_set='✓ Set' if YOUR_SECRET != 'your-secret-string' else '✗ Not set',
    groq_set='✓ Set' if GROQ_API_KEY != 'your-groq-api-key' else '✗ Not set'
))
    
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)