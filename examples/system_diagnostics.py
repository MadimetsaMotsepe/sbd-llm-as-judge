"""
System Diagnostics for LLM as Judge Service

This script provides comprehensive diagnostics including:
1. Basic API health checks and connectivity tests
2. Direct Azure OpenAI connectivity testing
3. Environment configuration validation
4. Database connectivity checks

Usage:
    python examples/system_diagnostics.py [--mode MODE]
    
Modes:
    - full: Run all diagnostics (default)
    - health: Run only basic health checks
    - azure: Run only Azure OpenAI connectivity tests
    - config: Run only configuration validation
"""

import argparse
import os
import asyncio
import requests
import json
from typing import Dict, Any, Optional

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import Azure OpenAI components (optional)
try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

BASE_URL = "http://localhost:8000"

# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================

def test_basic_endpoints():
    """Test basic API endpoints that don't require complex setup."""
    print("🔍 Testing Basic API Endpoints")
    print("=" * 50)
    
    results = []
    
    # Test 1: List judges (should work even if empty)
    try:
        response = requests.get(f"{BASE_URL}/list-judges", timeout=5)
        print(f"✅ GET /list-judges: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            judges_count = len(data.get('content', []))
            print(f"   Found {judges_count} judges in database")
            results.append(("list-judges", True, f"{judges_count} judges found"))
        else:
            print(f"   Response: {response.text[:200]}")
            results.append(("list-judges", False, f"Status {response.status_code}"))
    except Exception as e:
        print(f"❌ GET /list-judges failed: {e}")
        results.append(("list-judges", False, str(e)))
    
    # Test 2: List assemblies (should work even if empty)
    try:
        response = requests.get(f"{BASE_URL}/list-assemblies", timeout=5)
        print(f"✅ GET /list-assemblies: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            assemblies_count = len(data.get('content', []))
            print(f"   Found {assemblies_count} assemblies in database")
            results.append(("list-assemblies", True, f"{assemblies_count} assemblies found"))
        else:
            print(f"   Response: {response.text[:200]}")
            results.append(("list-assemblies", False, f"Status {response.status_code}"))
    except Exception as e:
        print(f"❌ GET /list-assemblies failed: {e}")
        results.append(("list-assemblies", False, str(e)))
    
    # Test 3: OpenAPI docs endpoint
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print(f"✅ GET /docs: {response.status_code}")
        results.append(("docs", response.status_code == 200, f"Status {response.status_code}"))
    except Exception as e:
        print(f"❌ GET /docs failed: {e}")
        results.append(("docs", False, str(e)))
    
    # Test 4: Health endpoint (if it exists)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ GET /health: {response.status_code}")
        if response.status_code == 200:
            print(f"   Health status: {response.text}")
        results.append(("health", response.status_code == 200, f"Status {response.status_code}"))
    except Exception as e:
        print(f"ℹ️  GET /health not available: {e}")
        results.append(("health", False, "Endpoint not available"))
    
    return results

def test_create_simple_judge():
    """Test creating a simple judge for validation."""
    print("\n🔧 Testing Judge Creation")
    print("=" * 50)
    
    simple_judge = {
        "id": "health-check-judge",
        "name": "HealthCheckJudge",
        "model": "https://sgndev01sbdoaiwu301.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview",
        "metaprompt": json.dumps({
            "text": "You are a simple health check judge. Just return 'OK' for any input.",
            "json": {"criteria": ["basic_check"], "scale": "OK|FAIL"}
        })
    }
    
    try:
        response = requests.post(f"{BASE_URL}/create-judge", 
                               json=simple_judge, 
                               headers={"Content-Type": "application/json"}, 
                               timeout=10)
        
        print(f"✅ POST /create-judge: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Judge created: {data.get('message', 'Success')}")
            return True, "Judge created successfully"
        else:
            print(f"   Error: {response.text[:200]}")
            return False, f"Status {response.status_code}: {response.text[:100]}"
            
    except Exception as e:
        print(f"❌ POST /create-judge failed: {e}")
        return False, str(e)

def test_simple_evaluation():
    """Test a simple evaluation to ensure the pipeline works."""
    print("\n⚡ Testing Simple Evaluation")
    print("=" * 50)
    
    eval_payload = {
        "id": "health-check-judge",
        "prompt": "This is a simple health check. Please respond with OK.",
        "method": "super"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/evaluate", 
                               json=eval_payload, 
                               headers={"Content-Type": "application/json"}, 
                               timeout=15)
        
        print(f"✅ POST /evaluate: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Evaluation result: {data.get('content', {}).get('result', 'No result')[:100]}...")
            return True, "Evaluation completed successfully"
        else:
            print(f"   Error: {response.text[:200]}")
            return False, f"Status {response.status_code}: {response.text[:100]}"
            
    except Exception as e:
        print(f"❌ POST /evaluate failed: {e}")
        return False, str(e)

def run_health_checks():
    """Run comprehensive health checks."""
    print("🏥 SYSTEM HEALTH CHECKS")
    print("=" * 60)
    
    # Test basic connectivity
    results = test_basic_endpoints()
    
    # Test judge creation
    judge_success, judge_msg = test_create_simple_judge()
    results.append(("create-judge", judge_success, judge_msg))
    
    # Test evaluation (only if judge creation succeeded)
    if judge_success:
        eval_success, eval_msg = test_simple_evaluation()
        results.append(("evaluation", eval_success, eval_msg))
    else:
        print("\n⚠️  Skipping evaluation test due to judge creation failure")
        results.append(("evaluation", False, "Skipped due to judge creation failure"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 HEALTH CHECK SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, details in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
    
    print(f"\nOverall Health: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 System is fully operational!")
    elif passed >= total * 0.7:
        print("⚠️  System is mostly operational with some issues")
    else:
        print("🚨 System has significant issues that need attention")
    
    return results

# =============================================================================
# AZURE OPENAI DIAGNOSTICS
# =============================================================================

async def test_azure_openai():
    """Test Azure OpenAI connectivity and configuration."""
    print("☁️  AZURE OPENAI DIAGNOSTICS")
    print("=" * 60)
    
    if not AZURE_AVAILABLE:
        print("❌ Azure OpenAI libraries not available")
        print("   Install with: pip install semantic-kernel")
        return False, "Libraries not available"
    
    # Get configuration
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4OMINI", "gpt-4o-mini")
    
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {version}")
    print(f"Key configured: {'Yes' if key else 'No'}")
    print(f"Key length: {len(key) if key else 0}")
    print()
    
    if not endpoint or not key:
        print("❌ Missing required environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_KEY")
        return False, "Missing configuration"
    
    try:
        # Create Azure OpenAI service
        print("Creating Azure OpenAI service...")
        azure_service = AzureChatCompletion(
            service_id=deployment,
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=key,
            api_version=version,
        )
        print("✅ Azure OpenAI service created successfully")
        
        # Test simple completion
        print("\nTesting simple completion...")
        chat_history = ChatHistory()
        chat_history.add_message(ChatMessageContent(role="user", content="Hello! Please respond with 'Connection successful'"))
        
        response = await azure_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=None
        )
        
        if response and len(response) > 0:
            result = response[0].content
            print(f"✅ Response received: {result[:100]}...")
            print(f"✅ Azure OpenAI connection successful!")
            return True, "Connection successful"
        else:
            print("❌ No response received from Azure OpenAI")
            return False, "No response received"
            
    except Exception as e:
        print(f"❌ Azure OpenAI test failed: {e}")
        return False, str(e)

def run_azure_diagnostics():
    """Run Azure OpenAI diagnostics."""
    try:
        success, message = asyncio.run(test_azure_openai())
        return success, message
    except Exception as e:
        print(f"❌ Azure diagnostics failed: {e}")
        return False, str(e)

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration():
    """Validate system configuration and environment variables."""
    print("⚙️  CONFIGURATION VALIDATION")
    print("=" * 60)
    
    config_items = [
        ("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"), "Azure OpenAI endpoint URL"),
        ("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY"), "Azure OpenAI API key"),
        ("AZURE_OPENAI_VERSION", os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview"), "Azure OpenAI API version"),
        ("AZURE_OPENAI_DEPLOYMENT_GPT4OMINI", os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4OMINI", "gpt-4o-mini"), "GPT-4o-mini deployment name"),
        ("USE_LOCAL_DB", os.getenv("USE_LOCAL_DB", "true"), "Use local JSON database"),
    ]
    
    results = []
    
    for name, value, description in config_items:
        if value:
            if "KEY" in name and len(value) > 10:
                display_value = f"{value[:10]}...({len(value)} chars)"
            else:
                display_value = value
            print(f"✅ {name}: {display_value}")
            results.append((name, True, display_value))
        else:
            print(f"❌ {name}: Not set")
            results.append((name, False, "Not set"))
        print(f"   {description}")
        print()
    
    # Check if local database file exists
    local_db_path = "src/local_db.json"
    if os.path.exists(local_db_path):
        try:
            with open(local_db_path, 'r') as f:
                db_content = json.load(f)
            judges_count = len(db_content.get('judges', []))
            assemblies_count = len(db_content.get('assemblies', []))
            print(f"✅ Local database: {local_db_path} exists")
            print(f"   Contains {judges_count} judges and {assemblies_count} assemblies")
            results.append(("local_database", True, f"{judges_count} judges, {assemblies_count} assemblies"))
        except Exception as e:
            print(f"⚠️  Local database exists but has issues: {e}")
            results.append(("local_database", False, f"File exists but corrupt: {e}"))
    else:
        print(f"ℹ️  Local database: {local_db_path} does not exist (will be created)")
        results.append(("local_database", True, "Will be created on first use"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)
    
    critical_configs = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"]
    critical_missing = [name for name, success, _ in results if name in critical_configs and not success]
    
    if critical_missing:
        print("🚨 Critical configuration missing:")
        for config in critical_missing:
            print(f"   - {config}")
        print("\nThe system will run in offline/mock mode only.")
    else:
        print("✅ All critical configuration present for Azure OpenAI connectivity")
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='System Diagnostics for LLM as Judge')
    parser.add_argument('--mode', choices=['full', 'health', 'azure', 'config'], 
                      default='full', help='Which diagnostics to run')
    
    args = parser.parse_args()
    
    print("🔧 LLM AS JUDGE - SYSTEM DIAGNOSTICS")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Base URL: {BASE_URL}")
    print("=" * 70)
    
    all_results = []
    
    if args.mode in ['full', 'config']:
        config_results = validate_configuration()
        all_results.extend(config_results)
        
    if args.mode in ['full', 'health']:
        if args.mode == 'full':
            print("\n")
        health_results = run_health_checks()
        all_results.extend([("health_" + name, success, details) for name, success, details in health_results])
        
    if args.mode in ['full', 'azure']:
        if args.mode == 'full':
            print("\n")
        azure_success, azure_message = run_azure_diagnostics()
        all_results.append(("azure_connectivity", azure_success, azure_message))
    
    # Overall summary
    if args.mode == 'full':
        print("\n" + "=" * 70)
        print("🏁 OVERALL SYSTEM STATUS")
        print("=" * 70)
        
        total_tests = len(all_results)
        passed_tests = sum(1 for _, success, _ in all_results if success)
        
        print(f"Total diagnostics: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 System is fully operational and ready for production!")
        elif passed_tests >= total_tests * 0.8:
            print("\n✅ System is operational with minor issues")
        elif passed_tests >= total_tests * 0.6:
            print("\n⚠️  System has some issues that should be addressed")
        else:
            print("\n🚨 System has significant issues requiring immediate attention")
    
    print("\n🏁 DIAGNOSTICS COMPLETE")

if __name__ == "__main__":
    main()
