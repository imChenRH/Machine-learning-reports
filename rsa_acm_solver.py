"""
RSA (Recursive Self-Aggregation) Algorithm for Solving ACM-ICPC Problems
Using Google Gemini API (gemini-1.5-pro model)

This program implements a workflow that uses LLM to solve complex algorithmic
problems through recursive decomposition and iterative refinement.

Author: [Your Name]
Course: Artificial Intelligence with Machine Learning
"""

import google.generativeai as genai
import subprocess
import sys
import re
import os
import tempfile
from typing import Optional


# Configure Gemini API - Load from environment variable for security
# Set your API key: export GEMINI_API_KEY="your_api_key_here"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)


class RSAAlgorithmSolver:
    """
    Implements the Recursive Self-Aggregation (RSA) algorithm to solve
    ACM-ICPC style programming problems using Google Gemini API (gemini-1.5-pro).
    """
    
    def __init__(self, max_iterations: int = 5):
        """
        Initialize the RSA solver.
        
        Args:
            max_iterations: Maximum number of refinement iterations
        """
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.max_iterations = max_iterations
        self.conversation_history = []
    
    def analyze_problem(self, problem_description: str) -> dict:
        """
        Step 1: Problem Decomposition
        Analyze the problem and break it into subproblems.
        
        Args:
            problem_description: The ACM-ICPC problem statement
            
        Returns:
            Dictionary containing problem analysis
        """
        prompt = f"""You are an expert competitive programmer. Analyze this ACM-ICPC problem 
and break it down into manageable subproblems.

Problem:
{problem_description}

Please provide:
1. Problem type (e.g., Dynamic Programming, Graph, Greedy, etc.)
2. Key observations and insights
3. List of subproblems to solve
4. Suggested algorithm approach
5. Time and space complexity considerations

Format your response clearly with labeled sections."""

        response = self.model.generate_content(prompt)
        analysis = response.text
        
        self.conversation_history.append({
            "role": "analysis",
            "content": analysis
        })
        
        return {
            "analysis": analysis,
            "problem": problem_description
        }
    
    def generate_solution(self, problem_analysis: dict) -> str:
        """
        Step 2: Solution Generation
        Generate a Python solution based on the problem analysis.
        
        Args:
            problem_analysis: The analysis from step 1
            
        Returns:
            Generated Python code as a string
        """
        prompt = f"""Based on the following problem analysis, generate a complete Python solution.

Problem:
{problem_analysis['problem']}

Analysis:
{problem_analysis['analysis']}

Requirements:
1. Write clean, efficient Python code
2. Include input/output handling (read from stdin, write to stdout)
3. Handle edge cases
4. Add brief comments explaining key parts
5. The code should be ready to submit to an online judge

Provide ONLY the Python code, no explanations outside the code."""

        response = self.model.generate_content(prompt)
        code = self._extract_code(response.text)
        
        self.conversation_history.append({
            "role": "solution",
            "content": code
        })
        
        return code
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code from markdown code blocks
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to find code without markers
        code_pattern = r'```\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Return the whole response if no code blocks found
        return response.strip()
    
    def test_solution(self, code: str, test_cases: list) -> dict:
        """
        Step 3: Solution Testing
        Test the generated solution against provided test cases.
        
        Args:
            code: The generated Python code
            test_cases: List of tuples (input_data, expected_output)
            
        Returns:
            Dictionary with test results
        """
        results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "details": []
        }
        
        for i, (input_data, expected_output) in enumerate(test_cases):
            try:
                # Write code to temporary file using tempfile for cross-platform compatibility
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                # Run the solution
                process = subprocess.run(
                    [sys.executable, temp_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                # Clean up temporary file
                os.unlink(temp_file)
                
                actual_output = process.stdout.strip()
                expected_output = expected_output.strip()
                
                if actual_output == expected_output:
                    results["passed"] += 1
                    results["details"].append({
                        "test": i + 1,
                        "status": "PASSED"
                    })
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "test": i + 1,
                        "input": input_data,
                        "expected": expected_output,
                        "actual": actual_output,
                        "stderr": process.stderr
                    })
                    results["details"].append({
                        "test": i + 1,
                        "status": "FAILED",
                        "error": f"Expected '{expected_output}', got '{actual_output}'"
                    })
                    
            except subprocess.TimeoutExpired:
                results["failed"] += 1
                results["errors"].append({
                    "test": i + 1,
                    "error": "Time Limit Exceeded"
                })
                results["details"].append({
                    "test": i + 1,
                    "status": "TLE"
                })
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "test": i + 1,
                    "error": str(e)
                })
                results["details"].append({
                    "test": i + 1,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        return results
    
    def refine_solution(self, code: str, test_results: dict, 
                        problem_analysis: dict) -> str:
        """
        Step 4: Recursive Refinement (RSA Core)
        Refine the solution based on test failures.
        
        Args:
            code: The current solution code
            test_results: Results from testing
            problem_analysis: Original problem analysis
            
        Returns:
            Refined Python code
        """
        if not test_results["errors"]:
            return code
        
        error_details = "\n".join([
            f"Test {e.get('test', '?')}: {e.get('error', 'Unknown error')}\n"
            f"  Input: {e.get('input', 'N/A')}\n"
            f"  Expected: {e.get('expected', 'N/A')}\n"
            f"  Got: {e.get('actual', 'N/A')}"
            for e in test_results["errors"]
        ])
        
        prompt = f"""The following solution has errors. Please fix them.

Original Problem:
{problem_analysis['problem']}

Current Code:
```python
{code}
```

Test Failures:
{error_details}

Previous Analysis:
{problem_analysis['analysis']}

Instructions:
1. Analyze why the tests are failing
2. Identify the bug or logical error
3. Provide a corrected version of the code
4. Make sure to handle all edge cases

Provide ONLY the corrected Python code, no explanations outside the code."""

        response = self.model.generate_content(prompt)
        refined_code = self._extract_code(response.text)
        
        self.conversation_history.append({
            "role": "refinement",
            "content": refined_code,
            "errors_fixed": error_details
        })
        
        return refined_code
    
    def solve(self, problem_description: str, test_cases: list) -> dict:
        """
        Main solving method implementing the full RSA workflow.
        
        Args:
            problem_description: The ACM-ICPC problem statement
            test_cases: List of (input, expected_output) tuples
            
        Returns:
            Dictionary with solution and process details
        """
        print("=" * 60)
        print("RSA Algorithm Solver - Starting")
        print("=" * 60)
        
        # Step 1: Analyze the problem
        print("\n[Step 1] Analyzing problem...")
        analysis = self.analyze_problem(problem_description)
        print("Analysis complete.")
        
        # Step 2: Generate initial solution
        print("\n[Step 2] Generating initial solution...")
        code = self.generate_solution(analysis)
        print("Initial solution generated.")
        
        # Iterative refinement loop (RSA core)
        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1}] Testing solution...")
            
            # Step 3: Test the solution
            results = self.test_solution(code, test_cases)
            
            print(f"  Passed: {results['passed']}/{len(test_cases)}")
            
            # Check if all tests pass
            if results["failed"] == 0:
                print("\n✓ All tests passed!")
                return {
                    "success": True,
                    "code": code,
                    "iterations": iteration + 1,
                    "analysis": analysis["analysis"],
                    "test_results": results
                }
            
            # Step 4: Refine the solution
            print(f"  Failed: {results['failed']} - Refining solution...")
            code = self.refine_solution(code, results, analysis)
        
        # Max iterations reached
        print(f"\n✗ Max iterations ({self.max_iterations}) reached.")
        final_results = self.test_solution(code, test_cases)
        
        return {
            "success": final_results["failed"] == 0,
            "code": code,
            "iterations": self.max_iterations,
            "analysis": analysis["analysis"],
            "test_results": final_results
        }


def main():
    """Example usage of the RSA Algorithm Solver."""
    
    # Example: Classic Two Sum Problem (simplified ACM-ICPC style)
    problem = """
    Two Sum Problem
    
    Given an array of integers nums and an integer target, find two numbers 
    such that they add up to target. Return their indices (0-based).
    
    You may assume that each input would have exactly one solution, 
    and you may not use the same element twice.
    
    Input Format:
    - First line: n (number of elements) and target
    - Second line: n space-separated integers
    
    Output Format:
    - Two space-separated indices
    
    Example:
    Input:
    4 9
    2 7 11 15
    
    Output:
    0 1
    
    Explanation: nums[0] + nums[1] = 2 + 7 = 9
    """
    
    # Test cases: (input, expected_output)
    test_cases = [
        ("4 9\n2 7 11 15", "0 1"),
        ("3 6\n3 2 4", "1 2"),
        ("2 6\n3 3", "0 1"),
    ]
    
    # Create solver and solve
    solver = RSAAlgorithmSolver(max_iterations=5)
    
    try:
        result = solver.solve(problem, test_cases)
        
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(f"Success: {result['success']}")
        print(f"Iterations: {result['iterations']}")
        print(f"\nFinal Code:\n{'-' * 40}")
        print(result['code'])
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure to set your GEMINI_API_KEY at the top of the file.")
        print("Get your API key from: https://makersuite.google.com/app/apikey")


if __name__ == "__main__":
    main()
