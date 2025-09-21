"""
Simple test to demonstrate content type detection logic
"""

def detect_content_type(content: str) -> str:
    """Detect content type based on keywords in the content."""
    content_lower = content.lower()
    
    # Database/Schema content
    if any(keyword in content_lower for keyword in ['database', 'schema', 'table', 'sql', 'entity']):
        return "Database/Schema"
    
    # Nintex/Workflow content  
    elif any(keyword in content_lower for keyword in ['nintex', 'workflow', 'automation', 'process', 'sharepoint']):
        return "Nintex/Workflow"
    
    # Application development
    elif any(keyword in content_lower for keyword in ['application', 'app', 'development', 'software', 'system']):
        return "Application Development"
    
    # Code-related content
    elif any(keyword in content_lower for keyword in ['code', 'function', 'class', 'programming', 'script', 'def ']):
        return "Code Review"
    
    # Business process content
    elif any(keyword in content_lower for keyword in ['business', 'requirement', 'process', 'procedure']):
        return "Business Process"
    
    # Default generic response
    else:
        return "Generic Content"

def main():
    """Test content type detection with different examples"""
    
    test_cases = [
        ("Create a database schema for employee onboarding with tables for employees, tasks, departments, and managers.", "Database/Schema"),
        ("Design a Nintex workflow for automating the employee onboarding process that integrates with SharePoint.", "Nintex/Workflow"),
        ("Develop a web application for tracking project milestones with user authentication and dashboard views.", "Application Development"),
        ("def calculate_benefits(salary, years): return salary * 0.1 if years > 5 else salary * 0.05", "Code Review"),
        ("Define the business requirements for the new customer onboarding process.", "Business Process"),
        ("Write a comprehensive guide on best practices for remote work productivity.", "Generic Content")
    ]
    
    print("Testing Content Type Detection:")
    print("=" * 50)
    
    for i, (content, expected) in enumerate(test_cases, 1):
        detected = detect_content_type(content)
        status = "✓" if detected == expected else "✗"
        
        print(f"{i}. Content: {content[:60]}...")
        print(f"   Expected: {expected}")
        print(f"   Detected: {detected} {status}")
        print()

if __name__ == "__main__":
    main()
