"""
Test script to verify the LLM-as-Judge system works for different content types
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app.judges import JudgeFactory, MockChatCompletion

async def test_different_content_types():
    """Test that mock responses adapt to different content types"""
    
    mock_service = MockChatCompletion("technical-judge")
    
    # Test database content
    database_content = """
    Create a database schema for employee onboarding with tables for employees, 
    tasks, departments, and managers. Include proper relationships and constraints.
    """
    
    # Test Nintex workflow content
    nintex_content = """
    Design a Nintex workflow for automating the employee onboarding process
    that integrates with SharePoint and sends notifications to managers.
    """
    
    # Test application development content  
    app_content = """
    Develop a web application for tracking project milestones with user authentication,
    dashboard views, and reporting capabilities using modern frameworks.
    """
    
    # Test code content
    code_content = """
    def calculate_employee_benefits(salary, years_service):
        if years_service < 1:
            return salary * 0.05
        elif years_service < 5:
            return salary * 0.10
        else:
            return salary * 0.15
    """
    
    print("Testing different content types with mock judge responses:\n")
    
    for content_type, content in [
        ("Database Schema", database_content),
        ("Nintex Workflow", nintex_content), 
        ("Application Development", app_content),
        ("Code Review", code_content)
    ]:
        print(f"=== {content_type} Content ===")
        response = mock_service._generate_adaptive_response(content)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(test_different_content_types())
