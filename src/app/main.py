import csv
import logging
import os
import json
import time
from typing import Optional, List, Dict, Any

from azure.cosmos import exceptions
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError, BaseModel

from app import __app__, __version__
from app.judges import JudgeOrchestrator, fetch_assembly
from app.schemas import RESPONSES, Assembly, ErrorMessage, Judge, JudgeEvaluation, SuccessMessage
from app.config import get_use_local_db, get_environment, get_log_level, GPTModel, get_model_args
from app.local_db import get_local_db
from openai import AzureOpenAI

from app.prompts import JUDGE_CREATION_PROMPT_TEMPLATE, ASSEMBLY_CREATION_PROMPT_TEMPLATE, IMPROVEMENT_PROMPT_TEMPLATE

load_dotenv(find_dotenv())

BLOB_CONN = os.getenv("BLOB_CONNECTION_STRING", "")
MODEL_URL: str = os.environ.get("GPT4_URL", "") # Assuming this is your base model URL for evaluation judges
MODEL_KEY: str = os.environ.get("GPT4_KEY", "")

MONITOR: str = os.environ.get("AZ_CONNECTION_LOG", "")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")

# Configuration
USE_LOCAL_DB = get_use_local_db()
ENVIRONMENT = get_environment()

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, get_log_level().upper(), logging.INFO))

# Get Azure OpenAI configuration
try:
    model_args = get_model_args(GPTModel.GPT4OMINI)
    AZURE_OPENAI_ENDPOINT = model_args.azure_openai_endpoint
    AZURE_OPENAI_API_KEY = model_args.azure_openai_key
    AZURE_OPENAI_DEPLOYMENT_NAME = model_args.deployment_id
    AZURE_OPENAI_VERSION = model_args.azure_openai_version
    
    # Log configuration for debugging
    logger.info(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Azure OpenAI Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    logger.info(f"Azure OpenAI Version: {AZURE_OPENAI_VERSION}")
    logger.info(f"Azure OpenAI Key configured: {'Yes' if AZURE_OPENAI_API_KEY else 'No'}")
    
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        logger.error("❌ Missing required Azure OpenAI environment variables:")
        logger.error("  - AZURE_OPENAI_ENDPOINT")
        logger.error("  - AZURE_OPENAI_KEY")
        raise ValueError("Missing Azure OpenAI configuration")
    
    # Initialize Azure OpenAI Client for Judge/Assembly creation
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_VERSION
    )
    logger.info("✅ Azure OpenAI client initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize Azure OpenAI client: {e}")
    raise

# Database helpers
async def get_database_client():
    """Get database client (local or Cosmos DB based on configuration)."""
    if USE_LOCAL_DB:
        return get_local_db()
    else:  # pragma: no cover - runtime branch
        return CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential())


tags_metadata: list[dict] = [
    {
        "name": "Management",
        "description": """
        Endpoints for managing the application and including eventual data for operation.
        """,
    },
    {
        "name": "Judge Configuration",
        "description": """
        Endpoints that allow the creation of agents, so the execution is stateless.
        """,
    },
    {
        "name": "Judge Execution",
        "description": """
        Endpoints to execute or trigger the judge evaluation.
        """,
    },
    {
        "name": "Automated Evaluation Suite",
        "description": """
        Endpoints for running comprehensive, automated LLM-as-Judge evaluation workflows.
        """,
    },
]

description: str = """
Web Service that manages the implementation of a full-fledged LLM as a Judge pattern. On this Pattern, the LLM is responsible for implementing custom evaluations
of tasks, and the orchestration of the LLM is done by a set of agents that are responsible for executing the tasks.
The service is responsible for managing the agents and the orchestration of the LLM, so it can be used as a standalone service or as a part of a larger system.
"""


app: FastAPI = FastAPI(
    title=__app__,
    version=__version__,
    description=description,
    openapi_tags=tags_metadata,
    openapi_url="/api/v1/openapi.json",
    responses=RESPONSES,  # type: ignore
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    validation_exception_handler Exception handler for validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Validation Error",
        title="Your request parameters didn't validate.",
        detail={"invalid-params": list(exc.errors())},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.exception_handler(ResponseValidationError)
async def response_exception_handler(
    request: Request, exc: ResponseValidationError  # pylint: disable=unused-argument
) -> JSONResponse:
    """
    response_exception_handler Exception handler for response validations.

    Args:
        request (Request): the request from the api
        exc (RequestValidationError): the validation raised by the process

    Returns:
        JSONResponse: A json encoded response with the validation errors.
    """

    response_body: ErrorMessage = ErrorMessage(
        success=False,
        type="Response Error",
        title="Found Errors on processing your requests.",
        detail={"invalid-params": list(exc.errors())},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_body),
    )


@app.get("/list-judges", tags=["Judge Configuration"])
async def list_judges(name: Optional[str] = None, email: Optional[str] = None) -> JSONResponse:
    """List all judges with optional name filtering."""
    if USE_LOCAL_DB:
        db = get_local_db()
        judges = await db.list_judges(name=name)
    else:  # pragma: no cover - requires cosmos
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
            try:
                database = client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                client.create_database(os.getenv("COSMOS_DB_NAME", ""))
            container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
            # Correct container alias is c
            query = "SELECT * FROM c"
            filters = []
            parameters = []
            if name:
                filters.append("CONTAINS(c.name, @judge_name)")
                parameters.append({"name": "@judge_name", "value": name})
            if email:
                filters.append("CONTAINS(c.email, @judge_email)")
                parameters.append({"name": "@judge_email", "value": email})
            if filters:
                query += " WHERE " + " AND ".join(filters)
            judges = [item async for item in container.query_items(query=query, parameters=parameters)]
    
    response_body: SuccessMessage = SuccessMessage(
        title=f"{len(judges)} Judges Retrieved",
        message="Successfully retrieved judge data from the database.",
        content=judges,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/create-judge", tags=["Judge Configuration"])
async def create_judge(judge: Judge) -> JSONResponse:
    """Create a new judge."""
    if USE_LOCAL_DB:
        db = get_local_db()
        created_judge = await db.create_judge(judge.model_dump())
    else:
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
            try:
                database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                cosmos_client.create_database(os.getenv("COSMOS_DB_NAME", ""))
            container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
            await container.upsert_item(judge.model_dump())
            created_judge = judge.model_dump()
    
    response_body: SuccessMessage = SuccessMessage(
        title=f"Judge {judge.name} Created",
        message="Judge created and ready for usage.",
        content=created_judge,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/update-judge/{judge_id}", tags=["Judge Configuration"])
async def update_judge(judge_id: str, judge: Judge) -> JSONResponse:
    """Update an existing judge (supports local DB when enabled)."""
    if USE_LOCAL_DB:
        db = get_local_db()
        try:
            updated_judge = await db.update_judge(judge_id, judge.model_dump())
        except KeyError:  # pragma: no cover - simple branch
            raise HTTPException(status_code=404, detail="Judge not found.")
    else:  # pragma: no cover - requires cosmos
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
            try:
                database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Database not found.")

            container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
            try:
                existing_client = await container.read_item(item=judge_id, partition_key=judge_id)
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Judge not found.")

            updated_judge = {**existing_client, **judge.model_dump()}
            await container.replace_item(item=judge_id, body=updated_judge)

    response_body: SuccessMessage = SuccessMessage(
        title=f"Judge {judge.name} Updated",
        message="Judge data has been updated successfully.",
        content=updated_judge,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/delete-judge/{judge_id}", tags=["Judge Configuration"])
async def delete_judge(judge_id: str) -> JSONResponse:  # type: ignore[override]
    """Delete a judge (supports local DB when enabled)."""
    if USE_LOCAL_DB:
        db = get_local_db()
        removed = await db.delete_judge(judge_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Judge not found.")
    else:  # pragma: no cover - requires cosmos
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
            try:
                database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Database not found.")

            container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
            try:
                await container.delete_item(item=judge_id, partition_key=judge_id)
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Judge not found.")

    response_body: SuccessMessage = SuccessMessage(
        title=f"Judge {judge_id} Deleted",
        message="Judge has been deleted successfully.",
        content={"judge_id": judge_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.get("/list-assemblies", tags=["Judge Configuration"])
async def list_assemblies(role: Optional[str] = None) -> JSONResponse:
    """List all assemblies with optional role filtering."""
    if USE_LOCAL_DB:
        db = get_local_db()
        assemblies = await db.list_assemblies(role=role)
    else:
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
            try:
                database = client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                client.create_database(os.getenv("COSMOS_DB_NAME", ""))
            container = database.get_container_client(os.getenv("COSMOS_ASSEMBLY_TABLE", ""))
            query = "SELECT * FROM c"
            parameters = []
            if role:
                query += " WHERE c.roles LIKE @role"
                parameters.append({"name": "@role", "value": f"%{role}%"})
            assemblies = [
                item async for item in container.query_items(query=query, parameters=parameters)
            ]
    
    response_body: SuccessMessage = SuccessMessage(
        title=f"{len(assemblies)} Assemblies Retrieved",
        message="Successfully retrieved assemblies with proper filter.",
        content=assemblies,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/create-assembly", tags=["Judge Configuration"])
async def create_assembly(assembly: Assembly) -> JSONResponse:
    """Create a new assembly."""
    if USE_LOCAL_DB:
        db = get_local_db()
        created_assembly = await db.create_assembly(assembly.model_dump())
    else:
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
            try:
                database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                cosmos_client.create_database(os.getenv("COSMOS_DB_NAME", ""))
            container = database.get_container_client(os.getenv("COSMOS_ASSEMBLY_TABLE", ""))
            await container.upsert_item(assembly.model_dump())
            created_assembly = assembly.model_dump()
    
    response_body: SuccessMessage = SuccessMessage(
        title="Assembly Created",
        message="Assembly has been created successfully.",
        content=created_assembly,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.put("/update-assembly/{assembly_id}", tags=["Judge Configuration"])
async def update_assembly(assembly_id: str, assembly: Assembly) -> JSONResponse:  # type: ignore[override]
    """Update an existing assembly (supports local DB when enabled)."""
    if USE_LOCAL_DB:
        db = get_local_db()
        try:
            updated_assembly = await db.update_assembly(assembly_id, assembly.model_dump())
        except KeyError:  # pragma: no cover
            raise HTTPException(status_code=404, detail="Assembly not found.")
    else:  # pragma: no cover - requires cosmos
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
            try:
                database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Database not found.")

            container = database.get_container_client(os.getenv("COSMOS_ASSEMBLY_TABLE", ""))
            try:
                existing_assembly = await container.read_item(
                    item=assembly_id, partition_key=assembly_id
                )
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Assembly not found.")

            updated_assembly = {**existing_assembly, **assembly.model_dump()}
            await container.replace_item(item=assembly_id, body=updated_assembly)

    response_body: SuccessMessage = SuccessMessage(
        title=f"Assembly {assembly_id} Updated",
        message="Assembly content has been updated successfully.",
        content=updated_assembly,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.delete("/delete-assembly/{assembly_id}", tags=["Judge Configuration"])
async def delete_assembly(assembly_id: str) -> JSONResponse:  # type: ignore[override]
    """Delete an assembly (supports local DB when enabled)."""
    if USE_LOCAL_DB:
        db = get_local_db()
        removed = await db.delete_assembly(assembly_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Assembly not found.")
    else:  # pragma: no cover - requires cosmos
        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
            try:
                database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                database.read()
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Database not found.")

            container = database.get_container_client(os.getenv("COSMOS_ASSEMBLY_TABLE", ""))
            try:
                await container.delete_item(item=assembly_id, partition_key=assembly_id)
            except exceptions.CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail="Assembly not found.")

    response_body: SuccessMessage = SuccessMessage(
        title=f"Assembly {assembly_id} Deleted",
        message="Assembly content has been deleted successfully.",
        content={"assembly_id": assembly_id},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))


@app.post("/evaluate", tags=["Judge Execution"])
async def evaluate_judgment(evaluation: JudgeEvaluation) -> JSONResponse:
    """
    Endpoint that evaluates a prompt using either:
      - An Assembly (evaluation.method == 'assembly')
      - A single Judge in "super" mode (evaluation.method == 'super'), wrapping it as a one-judge assembly.
    """
    db = get_local_db() if USE_LOCAL_DB else None

    # Determine source document
    if evaluation.method == "assembly":
        assembly_doc = await fetch_assembly(evaluation.id, db)
        if not assembly_doc:
            raise HTTPException(status_code=404, detail=f"Assembly '{evaluation.id}' not found.")
    else:  # super judge mode
        # Fetch single judge and wrap into assembly structure
        if USE_LOCAL_DB:
            judge_doc = await db.get_judge(evaluation.id)
        else:  # pragma: no cover - cosmos path
            async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:  # type: ignore[name-defined]
                try:
                    database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                    database.read()
                except exceptions.CosmosResourceNotFoundError:  # type: ignore[name-defined]
                    raise HTTPException(status_code=404, detail="Database not found.")
                container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
                try:
                    judge_doc = await container.read_item(item=evaluation.id, partition_key=evaluation.id)
                except exceptions.CosmosResourceNotFoundError:  # type: ignore[name-defined]
                    judge_doc = None
        if not judge_doc:
            raise HTTPException(status_code=404, detail=f"Judge '{evaluation.id}' not found.")
        assembly_doc = {
            "id": f"super-{evaluation.id}",
            "judges": [judge_doc],
            "roles": ["super"]
        }

    # Build pydantic Assembly (validates structure)
    try:
        assembly = Assembly(**assembly_doc)
    except ValidationError as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        final_verdict = await JudgeOrchestrator.run_evaluation(assembly, evaluation.prompt)
    except Exception as ex:  # pragma: no cover - broad fallback for runtime issues
        raise HTTPException(status_code=500, detail=str(ex)) from ex

    response_body = SuccessMessage(
        title="Evaluation Complete",
        message="Judging completed successfully.",
        content={"assembly_id": assembly.id, "result": final_verdict},
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))




class AutomatedEvaluationRequest(BaseModel):
    initial_user_query: str
    systemPrompt: str
    userPrompt: str
    input_requirements: str
    generated_output: Any
    structuredModel: Any
    run_improvement: bool = False
    evaluation_name: str = "Automated Evaluation"

@app.post("/run-full-evaluation-suite", tags=["Automated Evaluation Suite"])
async def run_full_evaluation_suite(request_data: AutomatedEvaluationRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    1. Dynamically creates judges based on input requirements and generated output using an LLM.
    2. Dynamically creates assemblies using an LLM based on the created judges.
    3. Runs evaluations with all created judges and assemblies.
    4. Optionally generates an improved output based on evaluation results.
    """
    logger.info(f"Starting full evaluation suite: {request_data.evaluation_name}")


    evaluation_suite_results: List[Dict[str, Any]] = []

    async def _process_full_evaluation():
        try:
            db = get_local_db() if USE_LOCAL_DB else None

            logger.info("Generating judges dynamically using LLM...")
            judge_prompt = JUDGE_CREATION_PROMPT_TEMPLATE.format(
                initial_user_query=request_data.initial_user_query,
                systemPrompt=request_data.systemPrompt,
                input_requirements=request_data.input_requirements,
                generated_output_snippet=request_data.generated_output
            )
            
            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": judge_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            judges_payload_raw = response.choices[0].message.content
            
            try:
                generated_judges_data = json.loads(judges_payload_raw)
                logger.debug(f"LLM returned judges JSON structure: {type(generated_judges_data)}, content: {generated_judges_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse judges JSON from LLM: {e}. Raw content: {judges_payload_raw}")
                raise HTTPException(status_code=500, detail=f"LLM failed to generate valid judge JSON: {e}")

            # Handle different JSON structures that the LLM might return
            try:
                if isinstance(generated_judges_data, dict):
                    # If the LLM returns an object with a key containing the list
                    if 'judges' in generated_judges_data:
                        generated_judges_list = generated_judges_data['judges']
                    elif 'data' in generated_judges_data:
                        generated_judges_list = generated_judges_data['data']
                    else:
                        # If it's a single judge object, wrap it in a list
                        generated_judges_list = [generated_judges_data]
                elif isinstance(generated_judges_data, list):
                    generated_judges_list = generated_judges_data
                elif isinstance(generated_judges_data, str):
                    # Handle case where JSON parsing returned a string somehow
                    logger.error(f"JSON parsing returned a string instead of object/list: {generated_judges_data}")
                    raise HTTPException(status_code=500, detail="LLM returned string instead of JSON object/array for judges")
                else:
                    logger.error(f"Unexpected judges JSON structure from LLM: {type(generated_judges_data)}. Content: {generated_judges_data}")
                    raise HTTPException(status_code=500, detail=f"LLM returned unexpected judges JSON structure: {type(generated_judges_data)}")
            except Exception as e:
                logger.error(f"Error processing judges data structure: {e}. Data type: {type(generated_judges_data)}, Content: {generated_judges_data}")
                raise HTTPException(status_code=500, detail=f"Failed to process judges data structure: {e}")

            created_judges_ids = []
            for judge_data in generated_judges_list:
                try:
                    # Validate that judge_data is a dictionary
                    if not isinstance(judge_data, dict):
                        logger.warning(f"Expected dictionary for judge data, got {type(judge_data)}: {judge_data}")
                        evaluation_suite_results.append({
                            "step": "Create Judge",
                            "judge_id": "N/A",
                            "status": "FAILED",
                            "error": f"Invalid judge data type: {type(judge_data)}"
                        })
                        continue
                    
                    # Truncate judge name if it's too long (max 32 characters)
                    if "name" in judge_data and len(judge_data["name"]) > 32:
                        original_name = judge_data["name"]
                        judge_data["name"] = judge_data["name"][:32]
                        logger.info(f"Truncated judge name from '{original_name}' to '{judge_data['name']}'")
                    
                    # Ensure the model URL is correct from env, as LLM might not add it.
                    judge_data["model"] = MODEL_URL # Override with your configured model URL
                    judge_obj = Judge(**judge_data)
                    # Call the existing create_judge logic
                    if USE_LOCAL_DB:
                        await db.create_judge(judge_obj.model_dump())
                    else:
                        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
                            database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                            container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
                            await container.upsert_item(judge_obj.model_dump())
                    created_judges_ids.append({"id": judge_obj.id, "name": judge_obj.name})
                    logger.info(f"Successfully created judge: {judge_obj.name}")
                except (ValidationError, Exception) as e:
                    judge_id = judge_data.get('id', 'N/A') if isinstance(judge_data, dict) else 'N/A'
                    logger.warning(f"Failed to create judge from LLM output ({judge_id}): {e}")
                    evaluation_suite_results.append({
                        "step": "Create Judge",
                        "judge_id": judge_id,
                        "status": "FAILED",
                        "error": str(e)
                    })

            if not created_judges_ids:
                raise HTTPException(status_code=500, detail="No judges were successfully created.")

            # --- Step 2: Dynamically Create Assemblies using LLM ---
            logger.info("Generating assemblies dynamically using LLM...")
            assembly_prompt = ASSEMBLY_CREATION_PROMPT_TEMPLATE.format(
                available_judges_info=json.dumps(created_judges_ids, indent=2)
            )

            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": assembly_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            assemblies_payload_raw = response.choices[0].message.content
            
            try:
                generated_assemblies_data = json.loads(assemblies_payload_raw)
                logger.debug(f"LLM returned assemblies JSON structure: {type(generated_assemblies_data)}, content: {generated_assemblies_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse assemblies JSON from LLM: {e}. Raw content: {assemblies_payload_raw}")
                raise HTTPException(status_code=500, detail=f"LLM failed to generate valid assembly JSON: {e}")

            # Handle different JSON structures that the LLM might return
            try:
                if isinstance(generated_assemblies_data, dict):
                    # If the LLM returns an object with a key containing the list
                    if 'assemblies' in generated_assemblies_data:
                        generated_assemblies_list = generated_assemblies_data['assemblies']
                    elif 'data' in generated_assemblies_data:
                        generated_assemblies_list = generated_assemblies_data['data']
                    else:
                        # If it's a single assembly object, wrap it in a list
                        generated_assemblies_list = [generated_assemblies_data]
                elif isinstance(generated_assemblies_data, list):
                    generated_assemblies_list = generated_assemblies_data
                elif isinstance(generated_assemblies_data, str):
                    # Handle case where JSON parsing returned a string somehow
                    logger.error(f"JSON parsing returned a string instead of object/list: {generated_assemblies_data}")
                    raise HTTPException(status_code=500, detail="LLM returned string instead of JSON object/array for assemblies")
                else:
                    logger.error(f"Unexpected assemblies JSON structure from LLM: {type(generated_assemblies_data)}. Content: {generated_assemblies_data}")
                    raise HTTPException(status_code=500, detail=f"LLM returned unexpected assemblies JSON structure: {type(generated_assemblies_data)}")
            except Exception as e:
                logger.error(f"Error processing assemblies data structure: {e}. Data type: {type(generated_assemblies_data)}, Content: {generated_assemblies_data}")
                raise HTTPException(status_code=500, detail=f"Failed to process assemblies data structure: {e}")

            created_assemblies_ids = []
            for assembly_data in generated_assemblies_list:
                try:
                    # Validate that assembly_data is a dictionary
                    if not isinstance(assembly_data, dict):
                        logger.warning(f"Expected dictionary for assembly data, got {type(assembly_data)}: {assembly_data}")
                        evaluation_suite_results.append({
                            "step": "Create Assembly",
                            "assembly_id": "N/A",
                            "status": "FAILED",
                            "error": f"Invalid assembly data type: {type(assembly_data)}"
                        })
                        continue
                    
                    # Truncate role names if they're too long (max 60 characters)
                    if "roles" in assembly_data and isinstance(assembly_data["roles"], list):
                        truncated_roles = []
                        for role in assembly_data["roles"]:
                            if isinstance(role, str) and len(role) > 60:
                                original_role = role
                                truncated_role = role[:60]
                                truncated_roles.append(truncated_role)
                                logger.info(f"Truncated assembly role from '{original_role}' to '{truncated_role}'")
                            else:
                                truncated_roles.append(role)
                        assembly_data["roles"] = truncated_roles
                    
                    # Validate that judges in assembly exist
                    for judge_id in assembly_data.get("judges", []):
                        if not any(j["id"] == judge_id for j in created_judges_ids):
                            raise ValueError(f"Judge ID '{judge_id}' not found among created judges.")

                    assembly_obj = Assembly(**assembly_data)
                    # Call the existing create_assembly logic
                    if USE_LOCAL_DB:
                        await db.create_assembly(assembly_obj.model_dump())
                    else:
                        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as cosmos_client:
                            database = cosmos_client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                            container = database.get_container_client(os.getenv("COSMOS_ASSEMBLY_TABLE", ""))
                            await container.upsert_item(assembly_obj.model_dump())
                    created_assemblies_ids.append(assembly_obj.id)
                    logger.info(f"Successfully created assembly: {assembly_obj.id}")
                except (ValidationError, Exception) as e:
                    assembly_id = assembly_data.get('id', 'N/A') if isinstance(assembly_data, dict) else 'N/A'
                    logger.warning(f"Failed to create assembly from LLM output ({assembly_id}): {e}")
                    evaluation_suite_results.append({
                        "step": "Create Assembly",
                        "assembly_id": assembly_id,
                        "status": "FAILED",
                        "error": str(e)
                    })
            
            if not created_assemblies_ids:
                raise HTTPException(status_code=500, detail="No assemblies were successfully created.")

            # Construct the evaluation prompt for judges/assemblies
            evaluation_prompt_content = f"""
EVALUATION CONTEXT:
Original User Query: {request_data.initial_user_query}

System Prompt: {request_data.systemPrompt}

INPUT REQUIREMENTS:
{request_data.input_requirements}

GENERATED OUTPUT:
{request_data.generated_output}

EVALUATION TASK:
Please evaluate the generated output against the input requirements from your domain expertise perspective. Consider how well the output:
1. Fulfills the stated requirements
2. Addresses the original user query intent  
3. Meets quality standards for this type of content
4. Demonstrates technical correctness and best practices
5. Provides value to the intended users/stakeholders

Provide your evaluation with specific observations, scores (where applicable), and actionable recommendations for improvement.
            """

            # --- Step 3: Run Individual Judge Evaluations ---
            logger.info("Running individual judge evaluations...")
            for judge_info in created_judges_ids:
                judge_id = judge_info["id"]
                judge_name = judge_info["name"]
                start_time = time.time()
                try:
                    # Create the evaluation payload for a single judge in 'super' mode
                    evaluation = JudgeEvaluation(id=judge_id, prompt=evaluation_prompt_content, method="super")
                    
                    # Fetch judge doc and wrap into an assembly structure for the orchestrator
                    if USE_LOCAL_DB:
                        judge_doc = await db.get_judge(judge_id)
                    else: # CosmosDB path
                        async with CosmosClient(COSMOS_ENDPOINT, DefaultAzureCredential()) as client:
                            database = client.get_database_client(os.getenv("COSMOS_DB_NAME", ""))
                            container = database.get_container_client(os.getenv("COSMOS_JUDGE_TABLE", ""))
                            judge_doc = await container.read_item(item=judge_id, partition_key=judge_id)
                    
                    if not judge_doc:
                        raise ValueError(f"Could not retrieve judge '{judge_id}' from DB.")
                    
                    assembly_doc = {"id": f"super-{judge_id}", "judges": [judge_id], "roles": ["super"]}
                    assembly = Assembly(**assembly_doc)

                    # Run the evaluation
                    verdict = await JudgeOrchestrator.run_evaluation(assembly, evaluation.prompt)
                    duration = time.time() - start_time
                    
                    evaluation_suite_results.append({
                        "step": "Individual Evaluation",
                        "judge_name": judge_name,
                        "status": "SUCCESS",
                        "duration_seconds": round(duration, 2),
                        "result": verdict
                    })
                    logger.info(f"Evaluation with judge '{judge_name}' completed successfully.")
                except Exception as e:
                    duration = time.time() - start_time
                    evaluation_suite_results.append({
                        "step": "Individual Evaluation",
                        "judge_name": judge_name,
                        "status": "FAILED",
                        "duration_seconds": round(duration, 2),
                        "error": str(e)
                    })
                    logger.error(f"Evaluation with judge '{judge_name}' failed: {e}")

            # --- Step 4: Run Assembly Evaluations ---
            logger.info("Running assembly evaluations...")
            for assembly_id in created_assemblies_ids:
                start_time = time.time()
                try:
                    evaluation = JudgeEvaluation(id=assembly_id, prompt=evaluation_prompt_content, method="assembly")
                    
                    # Fetch the full assembly document for the orchestrator
                    assembly_doc = await fetch_assembly(assembly_id)
                    if not assembly_doc:
                        raise ValueError(f"Could not retrieve assembly '{assembly_id}' from DB.")
                    
                    assembly = Assembly(**assembly_doc)
                    
                    verdict = await JudgeOrchestrator.run_evaluation(assembly, evaluation.prompt)
                    duration = time.time() - start_time
                    
                    evaluation_suite_results.append({
                        "step": "Assembly Evaluation",
                        "assembly_id": assembly_id,
                        "status": "SUCCESS",
                        "duration_seconds": round(duration, 2),
                        "result": verdict
                    })
                    logger.info(f"Evaluation with assembly '{assembly_id}' completed successfully.")
                except Exception as e:
                    duration = time.time() - start_time
                    evaluation_suite_results.append({
                        "step": "Assembly Evaluation",
                        "assembly_id": assembly_id,
                        "status": "FAILED",
                        "duration_seconds": round(duration, 2),
                        "error": str(e)
                    })
                    logger.error(f"Evaluation with assembly '{assembly_id}' failed: {e}")

            if request_data.run_improvement:
                logger.info("Generating improved output based on evaluation feedback...")
                feedback_summary = "\n".join(
                    f"Feedback from {res.get('judge_name') or res.get('assembly_id')}:\n{json.dumps(res.get('result'), indent=2)}\n---"
                    for res in evaluation_suite_results
                    if res.get('status') == 'SUCCESS' and res.get('result')
                )

                if feedback_summary:
                    improvement_prompt = IMPROVEMENT_PROMPT_TEMPLATE.format(
                        initial_user_query=request_data.initial_user_query,
                        original_input=request_data.input_requirements,
                        original_output=request_data.generated_output,
                        structuredModel=request_data.structuredModel,
                        evaluation_feedback=feedback_summary
                    )
                    
                    try:
                        response = openai_client.chat.completions.create(
                            model=AZURE_OPENAI_DEPLOYMENT_NAME,
                            messages=[
                                {"role": "system", "content": "You are a senior system architect specializing in refining technical designs based on expert feedback."},
                                {"role": "user", "content": improvement_prompt}
                            ]
                        )
                        improved_design = response.choices[0].message.content
                        evaluation_suite_results.append({
                            "step": "Improvement Generation",
                            "status": "SUCCESS",
                            "improved_output": improved_design
                        })
                        logger.info("Successfully generated improved output.")
                    except Exception as e:
                        logger.error(f"Failed to generate improved output: {e}")
                        evaluation_suite_results.append({
                            "step": "Improvement Generation",
                            "status": "FAILED",
                            "error": str(e)
                        })
                else:
                    logger.warning("Skipping improvement generation as there was no successful evaluation feedback.")

        except Exception as e:
            logger.error(f"An unexpected error occurred during the evaluation suite: {e}", exc_info=True)
            evaluation_suite_results.append({
                "step": "Overall Process",
                "status": "CRITICAL_FAILURE",
                "error": str(e)
            })
        finally:
            results_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
            os.makedirs(results_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{request_data.evaluation_name.replace(' ', '_')}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_suite_results, f, indent=4)
                logger.info(f"Evaluation suite '{request_data.evaluation_name}' complete. Results saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save evaluation results to file: {e}")

    background_tasks.add_task(_process_full_evaluation)

    response_body = SuccessMessage(
        title="Evaluation Suite Started",
        message=f"The automated evaluation suite '{request_data.evaluation_name}' has been started as a background task. Results will be saved to a file upon completion.",
        content={"evaluation_name": request_data.evaluation_name},
    )
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=jsonable_encoder(response_body))


