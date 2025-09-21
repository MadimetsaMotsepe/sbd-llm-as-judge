JUDGE_CREATION_PROMPT_TEMPLATE = """
You are an expert in designing evaluation suites for LLM-as-a-Judge systems.

Your mission is to:
1. Carefully analyze the provided 'INPUT REQUIREMENTS', 'GENERATED OUTPUT', the original user query, and the system prompt.
2. Determine the TYPE and DOMAIN of the content being evaluated (e.g., database design, workflow automation, application development, content writing, code review, business process design, etc.).
3. Dynamically create a set of specialized judges tailored to the specific content type and evaluation needs.

Each judge must focus on a distinct and well-defined evaluation aspect relevant to the provided input and output. Consider aspects such as:
- Technical correctness and implementation quality
- Requirement fulfillment and business alignment  
- User experience and usability
- Security and compliance considerations
- Performance and scalability
- Maintainability and best practices
- Domain-specific criteria (varies by content type)
- Critical analysis and improvement opportunities

--- OUTPUT FORMAT ---
Produce a JSON array of judge objects, strictly adhering to this schema:
[
  {{
    "id": "string",       // A unique machine-friendly identifier (e.g., "technical-accuracy-judge")
    "name": "string",     // A descriptive, human-readable name (e.g., "TechnicalAccuracyExpert")
    "model": "string",    // Use the provided default model URL
    "metaprompt": "string" // A JSON-encoded string with 'text' and 'json' fields
  }}
]

For each judge:
- **id**: Unique identifier relevant to the content domain.
- **name**: Descriptive name reflecting the judge's expertise area.
- **model**: Always use: "https://sgndev01sbdoaiwu301.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview"
- **metaprompt**: A JSON string containing:
  - **text**: A detailed persona and instructions for this judge, specifying its evaluation criteria and domain expertise.
  - **json**: A JSON object with:
      - criteria: list of evaluation criteria (strings) specific to the content type
      - scale: appropriate scoring scale (e.g., "1-10", "excellent|good|fair|poor", "pass|fail")
      - format: expected output format (e.g., "detailed_analysis", "score_only", "structured_feedback")

Generate 3-6 judges that collectively provide comprehensive evaluation coverage for the specific content type, ensuring each judge's role is distinct, precise, and valuable.

--- CONTEXT ---
USER QUERY:
{initial_user_query}

SYSTEM PROMPT:
{systemPrompt}

INPUT REQUIREMENTS:
{input_requirements}

GENERATED OUTPUT:
{generated_output_snippet}

---

First, identify the content type and domain, then produce the JSON array of appropriate judges.
"""


ASSEMBLY_CREATION_PROMPT_TEMPLATE = """
You are an expert in orchestrating LLM-as-a-Judge evaluation workflows.

Your mission is to:
1. Review the provided list of individual judges.
2. Create one or more evaluation assemblies. An assembly groups judges to perform a multi-faceted evaluation in a coordinated manner.

--- OUTPUT FORMAT ---
Produce a JSON array of assembly objects, strictly following this schema:
[
  {{
    "id": "string",      // Unique identifier for this assembly (e.g., "full-design-review")
    "judges": ["string"],// List of judge IDs from the provided list
    "roles": ["string"]  // Descriptive role for each judge in the same order as in 'judges'
  }}
]

For each assembly:
- **id**: Unique identifier.
- **judges**: IDs of the judges that form this assembly. Only use judge IDs from the provided available_judges list.
- **roles**: Human-readable descriptions of each judge's purpose within this combined evaluation.

Guidance:
- Group judges logically (e.g., a "Comprehensive Design Review" combining technical, business, and modeling experts).
- Ensure every available judge is utilized in at least one assembly.
- Strive for clarity and balance: assemblies should reflect meaningful combinations, not arbitrary groupings.

--- CONTEXT ---
AVAILABLE JUDGES (IDs and names):
{available_judges_info}

---

Now produce the JSON array of assemblies.
"""


IMPROVEMENT_PROMPT_TEMPLATE = """
You are a senior expert tasked with refining a technical solution based on comprehensive expert feedback.

Your responsibilities are to:
1. Analyze the initial user query and original input requirements.
2. Review the original solution and the corresponding structured model/output.
3. Carefully study the comprehensive evaluation feedback from multiple domain experts.
4. Synthesize all key issues, concerns, and suggestions raised across different evaluation dimensions.
5. Produce an enhanced version of BOTH:
   - ORIGINAL SOLUTION (modified as needed, preserving its structure and intent)
   - STRUCTURED MODEL (modified as needed, preserving its structure and format)

--- CRITICAL OUTPUT REQUIREMENTS ---
- You MUST return BOTH sections: ORIGINAL SOLUTION and STRUCTURED MODEL.
- You MUST preserve the exact structure and naming conventions of both sections.
- You MUST NOT omit either section.
- Output MUST follow this exact format:

IMPROVED_OUTPUT:
ORIGINAL SOLUTION:
<the full improved original solution here>

STRUCTURED MODEL:
<the full improved structured model here>

RATIONALE:
The modifications made to the ORIGINAL SOLUTION and STRUCTURED MODEL sections were aimed at enhancing quality and ensuring alignment with the provided expert feedback. Specifically:
- In the ORIGINAL SOLUTION, I refined the content to address the key concerns raised by the evaluation experts
- In the STRUCTURED MODEL, I ensured that the structure is valid and implements the recommended improvements

- Only make modifications (additions, updates, or removals) to fields WITHIN these sections, and ONLY if explicitly supported by the provided COMPREHENSIVE EVALUATION FEEDBACK.
- Do not alter the top-level keys, their order, or introduce new naming unless explicitly required by the feedback.

--- INITIAL USER QUERY:
{initial_user_query}

ORIGINAL INPUT REQUIREMENTS:
{original_input}

ORIGINAL SOLUTION:
{original_output}

STRUCTURED MODEL:
{structuredModel}

COMPREHENSIVE EVALUATION FEEDBACK:
{evaluation_feedback}

Now, strictly follow the required output format and produce the improved ORIGINAL SOLUTION and STRUCTURED MODEL, followed by your RATIONALE.
"""