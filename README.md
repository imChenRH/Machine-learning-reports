# Machine-learning-reports

This repository contains my personal course conclusion report on artificial intelligence and machine learning.

## Contents

- **RSA_Algorithm_LLM_ACM_ICPC_Paper.docx** - Personal course report on "Creating a Python Workflow Based on RSA (Recursive Self-Aggregation) Algorithm to Use LLM to Solve Hard Algorithm Problems from ACM-ICPC"
- **rsa_acm_solver.py** - Complete Python implementation using Google Gemini API
- **期末结课作业模板-paper版.docx** - Assignment template

## Report Sections

- Background and Motivation
- What I Learned
- My Implementation
- Results and Reflection
- References

## Python Program

The `rsa_acm_solver.py` program implements the RSA workflow using Google Gemini 3 Pro API. 

### Setup

1. Install dependencies:
   ```bash
   pip install google-generativeai
   ```

2. Get your Gemini API key from: https://makersuite.google.com/app/apikey

3. Set your API key as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

### Usage

```bash
python rsa_acm_solver.py
```

The program will:
1. Analyze the problem using LLM
2. Generate an initial solution
3. Test against provided test cases
4. Recursively refine the solution if tests fail

**Word count:** ~330 words (within 300-500 word requirement)