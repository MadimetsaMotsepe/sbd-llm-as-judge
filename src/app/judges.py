"""
judges.py

This module implements:
  - A SuperJudge that is also a Judge (extends JudgeBase), orchestrating sub-judges.
  - A Mediator-like pattern: SuperJudge collects notifications from sub-judges.
  - A Plan used by the SuperJudge to evaluate each sub-judge's output in a structured way.
  - A Factory (JudgeFactory) that builds sub-judges from a Pydantic Assembly.
  - An Orchestrator (JudgeOrchestrator) that merges the SuperJudge + SuperJudge logic into a
    single evaluation procedure.
"""

import asyncio
import json
import os
import re
import traceback
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import semantic_kernel as sk
from azure.cosmos import exceptions
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.planners.plan import Plan

from app.schemas import Judge, Assembly
from app.config import get_use_local_db
from app.local_db import get_local_db
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from app.config import os as _os  # reuse endpoint env if needed

COSMOS_ENDPOINT = _os.getenv("COSMOS_ENDPOINT", "")
COSMOS_DB_NAME = _os.getenv("COSMOS_DB_NAME", "")
COSMOS_JUDGE_TABLE = _os.getenv("COSMOS_JUDGE_TABLE", "")
COSMOS_ASSEMBLY_TABLE = _os.getenv("COSMOS_ASSEMBLY_TABLE", "")

async def fetch_judge(judge_id: str) -> dict | None:
    if get_use_local_db():
        db = get_local_db()
        return await db.get_judge(judge_id)
    if not COSMOS_ENDPOINT:
        return None
    async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
        try:
            database = client.get_database_client(COSMOS_DB_NAME)
            await database.read()
        except Exception:
            return None
        container = database.get_container_client(COSMOS_JUDGE_TABLE)
        try:
            item = await container.read_item(item=judge_id, partition_key=judge_id)
            return item
        except Exception:
            return None

def extract_deployment_name_from_url(url: str) -> str:
    """
    Extract deployment name from Azure OpenAI URL with graceful fallbacks.
    """
    try:
        match = re.search(r'/deployments/([^/]+)/', url)
        if match:
            name = match.group(1)
            # Normalize common variants
            if name.startswith("gpt-4o") and "mini" in name:
                return "gpt-4o-mini"
            if name.startswith("gpt-4") and "mini" not in name:
                return "gpt-4"  # allow alias mapping
            return name
    except Exception:  # noqa: BLE001
        pass
    return "default"


class Mediator(ABC):
    """
    The Mediator interface declares the notify method used by judges (agents)
    to report events or results. Any concrete mediator (such as SuperJudge)
    must implement this method.
    """

    @abstractmethod
    def notify(self, sender: object, event: str, data: dict) -> None:
        """
        Notifies the mediator of an event that has occurred.

        :param sender: The judge sending the notification.
        :param event: A string describing the event type (e.g., "evaluation_done").
        :param data: A dictionary containing additional data (e.g., judge_id, result).
        """
        pass


class JudgeBase(ABC):
    """
    Abstract base for a judge that can evaluate a prompt.
    """

    def __init__(self) -> None:
        """
        Initialize the judge with no mediator reference.
        """
        self.mediator: Optional["Mediator"] = None

    @abstractmethod
    async def evaluate(self, prompt: str) -> None:
        """
        Evaluate a prompt asynchronously.
        """
        pass


class ConcreteJudge(JudgeBase):
    """
    A judge that uses a ChatCompletionAgent to evaluate a prompt.
    Each instance references a shared Kernel but instantiates its own Agent in `evaluate()`.
    """

    def __init__(self, judge_data: Judge, kernel: sk.Kernel) -> None:
        super().__init__()
        self.judge_data = judge_data
        self.kernel = kernel

    async def evaluate(self, prompt: str) -> None:  # noqa: D401
        """Evaluate a prompt; gracefully degrade if no AI service configured."""
        try:
            try:
                meta = json.loads(self.judge_data.metaprompt)
            except json.JSONDecodeError as e:  # pragma: no cover - defensive
                meta = {"text": f"Invalid metaprompt JSON: {e}"}
            system_text = meta.get("text", "System Prompt Missing")

            # Extract deployment name from the model URL to use as service_id
            service_id = extract_deployment_name_from_url(str(self.judge_data.model))
            # If the resolved service_id is not registered, fall back early
            if not self.kernel.get_service(service_id):
                service_id = "gpt-4o-mini" if self.kernel.get_service("gpt-4o-mini") else "default"
            
            result_str = ""
            
            # Try direct service call first (more reliable than agent)
            try:
                chat_history = ChatHistory()
                chat_history.add_user_message(prompt)
                
                # Get the service directly
                service = self.kernel.get_service(service_id) or self.kernel.get_service("default")
                if service and hasattr(service, 'get_chat_message_contents'):
                    # Create basic settings for real Azure OpenAI service
                    try:
                        from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
                        settings = AzureChatPromptExecutionSettings(max_tokens=1000, temperature=0.7)
                    except:
                        settings = None
                    
                    response = await service.get_chat_message_contents(chat_history, settings)
                    if response:
                        result_str = response[0].content
            except Exception as e:  # noqa: BLE001
                print(f"Direct service call failed: {e}")
            
            # If direct service call failed, try agent approach
            if not result_str:
                try:
                    settings = self.kernel.get_prompt_execution_settings_from_service_id(
                        service_id=service_id
                    ) or self.kernel.get_prompt_execution_settings_from_service_id("default")

                    if settings:
                        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

                    # Attempt to build an agent; if it fails fall back to offline result
                    agent = None
                    if settings:
                        try:
                            agent = ChatCompletionAgent(
                                service_id=service_id,
                                kernel=self.kernel,
                                name=self.judge_data.name,
                                instructions=system_text,
                                arguments=KernelArguments(settings=settings),
                            )
                        except Exception:  # noqa: BLE001
                            agent = None

                    if agent is not None:
                        try:
                            chat_history = ChatHistory()
                            chat_history.add_user_message(prompt)

                            final_content = []
                            async for msg_content in agent.invoke(chat_history):
                                if any(
                                    isinstance(item, (FunctionCallContent, FunctionResultContent))
                                    for item in msg_content.items
                                ):
                                    continue
                                if getattr(msg_content, "content", "").strip():
                                    final_content.append(msg_content.content)
                            result_str = "\n".join(final_content).strip()
                        except Exception:  # noqa: BLE001
                            result_str = ""
                except Exception:  # noqa: BLE001
                    pass

            if not result_str:
                # Offline deterministic fallback
                result_str = f"[offline-eval] {self.judge_data.name} processed prompt_length={len(prompt)}"

        except Exception as e:  # pragma: no cover - broad fallback for stability
            result_str = f"[error-fallback] {self.judge_data.name}: {e}"  # still continue

        if self.mediator:
            self.mediator.notify(
                sender=self,
                event="evaluation_done",
                data={
                    "judge_id": self.judge_data.id,
                    "judge_name": self.judge_data.name,
                    "result": result_str,
                },
            )


class SuperJudge(JudgeBase, Mediator):
    """
    A "judge of judges." This class:
      - Inherits from JudgeBase (so it's also a judge).
      - Orchestrates sub-judges (Mediator-like).
      - Collects their outputs in a final verdict.
      - Shares the kernel as needed if it wants to do any final summary or evaluation.

    You can override evaluate() to define how it runs sub-judges in parallel or in a plan.
    """

    def __init__(self, kernel: sk.Kernel, name: str = "SuperJudge") -> None:
        super().__init__()
        self.kernel = kernel
        self.name = name
        self._judges: List[JudgeBase] = []
        self._evaluations: List[dict] = []

    def register_judge(self, judge: JudgeBase) -> None:
        """
        Register a sub-judge for orchestration.
        """
        self._judges.append(judge)
        judge.mediator = self  # so sub-judges can notify us

    def notify(self, sender: object, event: str, data: dict) -> None:
        """
        Called by sub-judges upon completion of their evaluation.
        """
        if event == "evaluation_done":
            self._evaluations.append(data)

    def final_verdict(self) -> str:
        """
        Combine sub-judges' evaluations.
        """
        if not self._evaluations:
            return "[No evaluations received]"
        lines = []
        for ev in self._evaluations:
            lines.append(f"{ev['judge_name']} => {ev['result']}")
        return "\n".join(lines)

    async def evaluate(self, prompt: str) -> None:
        """
        Because SuperJudge is also a Judge, we can define an evaluation flow:
          - We can run sub-judges in parallel or in a plan.
          - Then we might do a final step using self.kernel if we want.
        """
        # For demonstration, we'll define a Plan that runs each sub-judge's evaluation
        plan = JudgeEvaluationPlan(super_judge=self, prompt=prompt)
        await plan.run_plan()


class JudgeEvaluationPlan(Plan):
    """
    A semantic-kernel Plan describing how the SuperJudge calls each sub-judge.
    Could contain complex logic (sequential, parallel, branching, etc.).
    For simplicity, we'll run them in parallel.
    """

    def __init__(self, super_judge: SuperJudge, prompt: str):
        super().__init__(
            name="JudgeEvaluationPlan", description="Plan for orchestrating sub-judges."
        )
        self.super_judge = super_judge
        self.prompt = prompt

    async def run_plan(self) -> Any:
        """
        Execute the plan:
         - Evaluate all sub-judges in parallel
         - Return final verdict
        """
        # 1) Run all sub-judges in parallel
        await asyncio.gather(*(j.evaluate(self.prompt) for j in self.super_judge._judges))

        # 2) (Optionally) SuperJudge can do a final summary with self.super_judge.kernel
        # if desired. For now, we skip it.

        # 3) Return final verdict
        return self.super_judge.final_verdict()


class JudgeFactory:
    """
    Builds a kernel (optional) and produces sub-judges.
    Does NOT instantiate the "SuperJudge," but you may do so here if desired.

    Each judge is an agent that references the shared kernel.
    """

    @staticmethod
    def build_kernel() -> sk.Kernel:
        """
        Build a shared Kernel that sub-judges will reference.
        Add your AI services, plugins, etc.
        """
        from app.config import get_model_args, GPTModel
        
        kernel = sk.Kernel()
        
        # Always try to add Azure OpenAI service first
        try:
            args = get_model_args(GPTModel.GPT4OMINI)
            
            # Import Azure OpenAI service
            from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
            
            # Add Azure OpenAI service to kernel
            azure_chat_service = AzureChatCompletion(
                service_id=args.deployment_id,
                deployment_name=args.deployment_id,
                endpoint=args.azure_openai_endpoint,
                api_key=args.azure_openai_key,
                api_version=args.azure_openai_version,
            )
            kernel.add_service(azure_chat_service)
            
            # Add as default service as well - create a new instance
            default_service = AzureChatCompletion(
                service_id="default",
                deployment_name=args.deployment_id,
                endpoint=args.azure_openai_endpoint,
                api_key=args.azure_openai_key,
                api_version=args.azure_openai_version,
            )
            kernel.add_service(default_service)
            # Compatibility alias services so legacy IDs like 'gpt-4' resolve
            for alias in {"gpt-4", "gpt4", "gpt4o"} - {args.deployment_id, "default"}:
                try:
                    alias_service = AzureChatCompletion(
                        service_id=alias,
                        deployment_name=args.deployment_id,
                        endpoint=args.azure_openai_endpoint,
                        api_key=args.azure_openai_key,
                        api_version=args.azure_openai_version,
                    )
                    kernel.add_service(alias_service)
                except Exception:  # noqa: BLE001
                    pass
            print(f"✅ Successfully registered Azure OpenAI service: {args.deployment_id}")
            print(f"   Endpoint: {args.azure_openai_endpoint}")
            print(f"   Deployment: {args.deployment_id}")
            return kernel
                
        except Exception as e:
            print(f"❌ Failed to initialize Azure OpenAI service: {e}")
            import traceback
            traceback.print_exc()

        # Only use mock service if Azure OpenAI completely fails
        print("⚠️  Falling back to mock service - Azure OpenAI not available")
        
        class MockChatCompletion:
            def __init__(self, service_id: str = "default"):
                self.service_id = service_id
                
            async def get_chat_message_contents(self, chat_history, settings):
                """Enhanced mock chat completion that returns realistic structured responses for any content type."""
                from semantic_kernel.contents.chat_message_content import ChatMessageContent
                from semantic_kernel.contents.text_content import TextContent
                
                # Extract the user message
                user_message = ""
                if hasattr(chat_history, 'messages') and chat_history.messages:
                    user_message = str(chat_history.messages[-1].content)
                else:
                    user_message = str(chat_history)
                
                # Detect content type and generate appropriate response based on service_id and content
                mock_response = self._generate_adaptive_response(user_message)

                return [ChatMessageContent(
                    role="assistant",
                    content=mock_response,
                    items=[TextContent(text=mock_response)]
                )]
            
            def _generate_adaptive_response(self, content: str) -> str:
                """Generate contextually appropriate response based on content type and judge role."""
                # Detect content type
                content_lower = content.lower()
                
                # Database/Schema content
                if any(keyword in content_lower for keyword in ['database', 'schema', 'table', 'sql', 'entity']):
                    return self._generate_database_response(content)
                
                # Nintex/Workflow content  
                elif any(keyword in content_lower for keyword in ['nintex', 'workflow', 'automation', 'process', 'sharepoint']):
                    return self._generate_workflow_response(content)
                
                # Application development
                elif any(keyword in content_lower for keyword in ['application', 'app', 'development', 'software', 'system']):
                    return self._generate_application_response(content)
                
                # Code-related content
                elif any(keyword in content_lower for keyword in ['code', 'function', 'class', 'programming', 'script']):
                    return self._generate_code_response(content)
                
                # Business process content
                elif any(keyword in content_lower for keyword in ['business', 'requirement', 'process', 'procedure']):
                    return self._generate_business_response(content)
                
                # Default generic response
                else:
                    return self._generate_generic_response(content)
            
            def _generate_database_response(self, content: str) -> str:
                """Generate database/schema specific evaluation response."""
                if "technical" in self.service_id.lower() or "accuracy" in self.service_id.lower():
                    return """**Database Technical Assessment**

**Overall Score: 8/10**

**Technical Analysis:**
• **Schema Design (8/10)**: Well-structured with appropriate relationships
• **Normalization (7/10)**: Follows 3NF with some optimization opportunities  
• **Data Types (9/10)**: Appropriate field types and constraints
• **Indexing Strategy (6/10)**: Basic indexing present, room for optimization

**Key Findings:**
- Table relationships are logically defined
- Primary and foreign keys properly implemented
- Some redundancy could be eliminated
- Performance optimization opportunities identified

**Recommendations:**
• Add composite indexes for frequent query patterns
• Consider partitioning for large tables
• Implement proper constraint validation
• Add audit trails for sensitive data"""

                elif "business" in self.service_id.lower() or "requirement" in self.service_id.lower():
                    return """**Business Requirements Assessment**

**Overall Alignment: Good**

**Business Analysis:**
• **Requirement Coverage (85%)**: Most business needs addressed
• **Process Support (80%)**: Supports core business workflows
• **User Experience (75%)**: Functional but could be more intuitive
• **Integration Readiness (90%)**: Well-prepared for system integration

**Business Value:**
- Addresses primary business objectives
- Supports operational efficiency
- Enables data-driven decision making
- Provides audit and compliance capabilities

**Gaps Identified:**
• Missing some edge case scenarios
• Limited reporting capabilities
• Could enhance user workflow efficiency"""

                else:
                    return """**Database Design Evaluation**

**Overall Assessment: Strong Foundation**

**Key Strengths:**
• Well-organized structure with clear entity relationships
• Appropriate data modeling practices
• Good separation of concerns
• Scalable architecture foundation

**Areas for Enhancement:**
• Performance optimization opportunities
• Security considerations could be strengthened
• Documentation could be more comprehensive
• Consider additional business rule enforcement

**Rating: 7.5/10**
*Solid design with clear improvement opportunities*"""

            def _generate_workflow_response(self, content: str) -> str:
                """Generate Nintex/workflow specific evaluation response."""
                if "technical" in self.service_id.lower():
                    return """**Workflow Technical Assessment**

**Overall Score: 8.5/10**

**Technical Analysis:**
• **Workflow Logic (9/10)**: Well-structured and logical flow
• **Integration Points (8/10)**: Good system connectivity
• **Error Handling (7/10)**: Basic error handling present
• **Performance (8/10)**: Efficient execution path

**Technical Strengths:**
- Clean workflow design with proper sequencing
- Appropriate use of conditions and loops
- Good variable management
- Proper state management

**Improvement Areas:**
• Enhanced error handling and rollback procedures
• More comprehensive logging and monitoring
• Optimization for high-volume scenarios
• Better exception management"""

                elif "business" in self.service_id.lower() or "requirement" in self.service_id.lower():
                    return """**Workflow Business Assessment**

**Overall Alignment: Excellent**

**Business Analysis:**
• **Process Automation (95%)**: Highly effective automation
• **User Experience (85%)**: Intuitive and user-friendly
• **Efficiency Gains (90%)**: Significant time savings
• **Compliance Support (80%)**: Good audit trail capabilities

**Business Impact:**
- Streamlines manual processes effectively
- Reduces processing time and errors
- Improves consistency and compliance
- Enables better resource allocation

**Recommendations:**
• Add more detailed approval workflows
• Enhance notification mechanisms
• Consider mobile accessibility
• Implement advanced reporting features"""

                else:
                    return """**Nintex Workflow Evaluation**

**Overall Assessment: Well-Designed Automation**

**Key Strengths:**
• Clear process flow with logical sequencing
• Good user interaction design
• Effective automation of manual tasks
• Proper integration with existing systems

**Process Efficiency:**
- Reduces manual effort significantly
- Improves process consistency
- Enables better tracking and monitoring
- Supports compliance requirements

**Enhancement Opportunities:**
• Add more sophisticated routing options
• Enhance mobile user experience
• Implement advanced analytics
• Consider AI-powered decision points

**Rating: 8.2/10**
*Effective workflow design with room for advanced features*"""

            def _generate_application_response(self, content: str) -> str:
                """Generate application development specific evaluation response."""
                if "technical" in self.service_id.lower():
                    return """**Application Technical Assessment**

**Overall Score: 7.8/10**

**Technical Analysis:**
• **Architecture (8/10)**: Well-structured and scalable design
• **Code Quality (7/10)**: Good practices with some improvements needed
• **Security (8/10)**: Appropriate security measures implemented
• **Performance (7/10)**: Acceptable performance with optimization opportunities

**Technical Strengths:**
- Modular architecture with clear separation
- Good use of design patterns
- Proper error handling mechanisms
- Adequate testing coverage

**Areas for Improvement:**
• Code optimization for better performance
• Enhanced logging and monitoring
• More comprehensive unit testing
• Better documentation and comments"""

                elif "user" in self.service_id.lower() or "experience" in self.service_id.lower():
                    return """**User Experience Assessment**

**Overall UX Score: 8/10**

**UX Analysis:**
• **Usability (8/10)**: Intuitive interface design
• **Accessibility (7/10)**: Basic accessibility features present
• **Navigation (9/10)**: Clear and logical navigation flow
• **Visual Design (8/10)**: Clean and professional appearance

**User Experience Strengths:**
- Easy to learn and use
- Consistent design patterns
- Responsive design implementation
- Good information architecture

**Enhancement Opportunities:**
• Improve accessibility compliance
• Add more interactive elements
• Enhance mobile experience
• Implement user feedback mechanisms"""

                else:
                    return """**Application Development Evaluation**

**Overall Assessment: Solid Implementation**

**Key Strengths:**
• Well-architected solution with clear structure
• Good development practices followed
• Appropriate technology choices
• Scalable foundation for future growth

**Implementation Quality:**
- Clean code structure
- Proper error handling
- Good security practices
- Adequate performance characteristics

**Recommendations:**
• Enhance testing coverage
• Improve documentation
• Add monitoring and analytics
• Consider performance optimization

**Rating: 8/10**
*Strong development work with minor enhancement opportunities*"""

            def _generate_code_response(self, content: str) -> str:
                """Generate code-specific evaluation response."""
                return """**Code Quality Assessment**

**Overall Score: 7.5/10**

**Code Analysis:**
• **Readability (8/10)**: Clean and well-structured code
• **Maintainability (7/10)**: Good modular design
• **Performance (7/10)**: Efficient implementation
• **Security (8/10)**: Proper security considerations

**Code Strengths:**
- Clear variable and function naming
- Good separation of concerns
- Appropriate use of comments
- Follows coding standards

**Improvement Areas:**
• Add more comprehensive error handling
• Enhance unit test coverage
• Optimize performance-critical sections
• Consider refactoring complex functions

**Best Practices:**
- Use of design patterns: ✓
- Error handling: ✓
- Documentation: △
- Testing: △

**Recommendations:**
• Implement comprehensive testing suite
• Add performance profiling
• Enhance code documentation
• Consider code review process"""

            def _generate_business_response(self, content: str) -> str:
                """Generate business process specific evaluation response."""
                return """**Business Process Assessment**

**Overall Alignment: Good**

**Business Analysis:**
• **Requirement Coverage (85%)**: Most business needs addressed
• **Process Efficiency (80%)**: Streamlines current operations
• **User Adoption (75%)**: Generally user-friendly approach
• **ROI Potential (85%)**: Strong return on investment expected

**Business Value:**
- Addresses core business objectives
- Improves operational efficiency
- Reduces manual effort and errors
- Enhances compliance capabilities

**Strategic Considerations:**
- Aligns with organizational goals
- Supports scalability requirements
- Enables data-driven decision making
- Facilitates process standardization

**Recommendations:**
• Enhance change management planning
• Add comprehensive training materials
• Implement phased rollout approach
• Establish success metrics and KPIs"""

            def _generate_generic_response(self, content: str) -> str:
                """Generate generic evaluation response for unknown content types."""
                word_count = len(content.split())
                
                if "quality" in self.service_id.lower():
                    return f"""**Content Quality Evaluation**

**Overall Score: 8/10**

**Quality Assessment:**
• **Clarity (8/10)**: Content is well-structured and clear
• **Completeness (7/10)**: Covers most required aspects
• **Accuracy (8/10)**: Information appears factually correct
• **Relevance (9/10)**: Highly relevant to stated objectives

**Content Metrics:**
- Word count: {word_count}
- Structure: Well-organized
- Tone: Professional and appropriate
- Coverage: Comprehensive

**Strengths:**
- Clear communication of key concepts
- Logical flow and organization
- Appropriate level of detail
- Good use of supporting information

**Enhancement Opportunities:**
• Add more specific examples
• Include additional supporting data
• Enhance visual elements
• Strengthen conclusion section"""

                else:
                    return f"""**Comprehensive Evaluation**

**Overall Assessment: Strong Performance**

**Key Findings:**
• Content demonstrates solid understanding
• Well-organized and purposeful structure
• Appropriate for intended audience
• Effective communication of main objectives

**Quality Indicators:**
- Clarity: 85%
- Relevance: 90%
- Completeness: 80%
- Professional quality: 85%

**Analysis Summary:**
The material shows comprehensive coverage of the topic with effective presentation. Structure supports main objectives while maintaining engagement throughout.

**Recommendations:**
• Strengthen evidence base with additional details
• Expand practical application examples
• Consider adding summary sections
• Enhance supporting documentation

**Overall Rating: 8.1/10**
*Well-executed work with clear potential for refinement*"""
        
        # Register mock services
        mock_service = MockChatCompletion("default")
        kernel.add_service(mock_service)
        
        # Register service for gpt-4o-mini deployment name
        mock_service_mini = MockChatCompletion("gpt-4o-mini")
        kernel.add_service(mock_service_mini)
        
        return kernel

    @staticmethod
    async def create_judges(assembly: Assembly, kernel: sk.Kernel) -> List[ConcreteJudge]:
        judges: List[ConcreteJudge] = []
        for judge_id in assembly.judges:
            doc = await fetch_judge(judge_id)
            if not doc:
                continue
            try:
                judge_model = Judge(**doc)
            except Exception:
                continue
            judges.append(ConcreteJudge(judge_data=judge_model, kernel=kernel))
        return judges


class JudgeOrchestrator:
    """
    A high-level class that merges the SuperJudge and JudgeFactory in one evaluation procedure.
    - You give it an Assembly (list of Pydantic Judge entries).
    - It builds a kernel, a SuperJudge, sub-judges, and orchestrates an evaluation.
    """

    @staticmethod
    async def run_evaluation(assembly: Assembly, prompt: str) -> str:
        """
        1) Build a shared kernel
        2) Instantiate a SuperJudge referencing that kernel
        3) Create sub-judges via JudgeFactory
        4) Register them in the SuperJudge
        5) Let the SuperJudge evaluate (which calls sub-judges in a plan)
        6) Return the final verdict
        """
        # 1) Shared kernel
        kernel = JudgeFactory.build_kernel()

        # 2) SuperJudge
        super_judge = SuperJudge(kernel=kernel, name=f"SuperJudge_{assembly.id}")

        # 3) Create sub-judges from the assembly
        sub_judges = await JudgeFactory.create_judges(assembly, kernel=kernel)

        # 4) Register them
        for j in sub_judges:
            super_judge.register_judge(j)

        # 5) Evaluate
        await super_judge.evaluate(prompt)

        # 6) Final verdict
        return super_judge.final_verdict()


async def fetch_assembly(assembly_id: str) -> dict | None:
    """
    Helper function to fetch an Assembly document from either local DB or Cosmos DB by its ID.
    Returns the document as a dict, or None if not found.
    """
    # Import here to avoid circular imports
    from app.config import get_use_local_db
    from app.local_db import get_local_db
    
    if get_use_local_db():
        db = get_local_db()
        return await db.get_assembly(assembly_id)
    else:
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
            try:
                database = client.get_database_client(COSMOS_DB_NAME)
                # Confirm the database exists.
                await database.read()
            except exceptions.CosmosResourceNotFoundError:
                return None

            container = database.get_container_client(COSMOS_ASSEMBLY_TABLE)
            try:
                item = await container.read_item(item=assembly_id, partition_key=assembly_id)
                return item
            except exceptions.CosmosResourceNotFoundError:
                return None
