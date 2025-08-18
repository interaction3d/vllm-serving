import os
import asyncio
import time
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from google.cloud import storage

from vllm import LLM, SamplingParams

# Minimal OpenTelemetry imports for Google Cloud Trace
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)


os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

load_dotenv()

# Initialize Google Cloud Trace
def setup_tracing():
    # Get project ID for tracing
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "image-classification-terraform")
    
    # Set up the tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer_provider()
    
    # Create Cloud Trace exporter
    cloud_trace_exporter = CloudTraceSpanExporter(project_id=project_id)
    
    # Add the exporter to the tracer provider
    tracer.add_span_processor(SimpleSpanProcessor(cloud_trace_exporter))
    
    print(f"Google Cloud Trace initialized for project: {project_id}")


def upload_profile_to_gcs():
    """Upload vLLM profiling data to Google Cloud Storage."""
    try:
        profile_dir = Path("./vllm_profile")
        if not profile_dir.exists():
            print("No profile directory found, skipping upload")
            return
        
        # Initialize GCS client
        client = storage.Client()
        bucket_name = "torch-trace-output"
        bucket = client.bucket(bucket_name)
        
        # Upload all files recursively
        uploaded_count = 0
        for file_path in profile_dir.rglob("*"):
            if file_path.is_file():
                # Create relative path for GCS object name
                relative_path = file_path.relative_to(".")
                blob_name = str(relative_path)
                
                # Upload file
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                uploaded_count += 1
                print(f"Uploaded: {blob_name}")
        
        print(f"Successfully uploaded {uploaded_count} profile files to gs://{bucket_name}/")
        
    except Exception as e:
        print(f"Warning: Failed to upload profile data to GCS: {e}")

# Setup tracing before creating FastAPI app
setup_tracing()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-20b")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
TP_DEGREE = int(os.getenv("TP_DEGREE", "1"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes"}

# bug when loading model from huggingface
# https://github.com/vllm-project/vllm/issues/22383
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text")
    temperature: float = 0.7
    max_tokens: int = Field(256, ge=1, le=8192)
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    text: str
    num_tokens: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = Field(256, ge=1, le=8192)
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None


class ChatResponse(BaseModel):
    text: str
    num_tokens: int


app = FastAPI(title="vLLM FastAPI Server", version="0.1.0")


_llm: Optional[LLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def instrument_llm_layers(llm: LLM):
    """Instrument LLM model layers with OpenTelemetry spans for detailed tracing."""
    tracer = trace.get_tracer(__name__)
    
    def otel_layer_hook(module, input, output):
        """Hook function to create spans for each layer execution."""
        layer_name = module.__class__.__name__
        with tracer.start_as_current_span(f"layer.{layer_name}"):
            pass  # The actual layer execution happens automatically
    
    try:
        # Access the underlying PyTorch model
        torch_model = llm.model
        
        # Register hooks for transformer layers
        layer_count = 0
        for name, module in torch_model.named_modules():
            module.register_forward_hook(otel_layer_hook)
            layer_count += 1
        
        print(f"Instrumented {layer_count} model layers with OpenTelemetry spans")
        
    except Exception as e:
        print(f"Warning: Could not instrument model layers: {e}")


async def get_llm() -> LLM:
    global _llm
    if _llm is None:
        def _init_llm() -> LLM:
            llm = LLM(
                model=MODEL_NAME,
                tensor_parallel_size=TP_DEGREE,
                trust_remote_code=TRUST_REMOTE_CODE,
                max_model_len=MAX_MODEL_LEN,
            )
            
            # Instrument the model layers for detailed tracing
            instrument_llm_layers(llm)
            
            return llm

        loop = asyncio.get_running_loop()
        _llm = await loop.run_in_executor(None, _init_llm)
    return _llm


async def get_tokenizer() -> AutoTokenizer:
    print("Getting tokenizer...")
    global _tokenizer
    if _tokenizer is None:
        def _init_tokenizer() -> AutoTokenizer:
            kwargs = dict(trust_remote_code=TRUST_REMOTE_CODE)
            return AutoTokenizer.from_pretrained(MODEL_NAME, **kwargs)

        loop = asyncio.get_running_loop()
        _tokenizer = await loop.run_in_executor(None, _init_tokenizer)
    return _tokenizer


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    tracer = trace.get_tracer(__name__)
    start_time = time.time()
    
    with tracer.start_as_current_span("generate") as span:
        try:
            with tracer.start_as_current_span("generate.sampling_params"):
                sampling = SamplingParams(
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    presence_penalty=req.presence_penalty,
                    frequency_penalty=req.frequency_penalty,
                    stop=req.stop,
                )

            with tracer.start_as_current_span("generate.get_model"):
                model = await get_llm()
            
            model.start_profile()
            with tracer.start_as_current_span("generate.outputs"):
                # outputs = await asyncio.get_running_loop().run_in_executor(
                #     None, lambda: model.generate(prompts=[req.prompt], sampling_params=sampling)
                # )
                outputs = model.generate(prompts=[req.prompt], sampling_params=sampling)
            model.stop_profile()
            
            # Upload profile data to GCS
            upload_profile_to_gcs()
            
            output = outputs[0]
            text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            
            # Add basic metrics to span
            duration = time.time() - start_time
            span.set_attribute("duration_ms", int(duration * 1000))
            span.set_attribute("num_tokens", num_tokens)
            span.set_attribute("max_tokens", req.max_tokens)
            
            return GenerateResponse(text=text, num_tokens=num_tokens)
        except Exception as e:
            span.set_attribute("error", True)
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    tracer = trace.get_tracer(__name__)
    start_time = time.time()
    
    with tracer.start_as_current_span("chat") as span:
        try:
            # Convert messages to the format expected by apply_chat_template
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            
            # Use the model's built-in chat template
            tokenizer = await get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            sampling = SamplingParams(
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
                top_k=req.top_k,
                presence_penalty=req.presence_penalty,
                frequency_penalty=req.frequency_penalty,
                stop=req.stop,
            )
            
            model = await get_llm()
            outputs = await asyncio.get_running_loop().run_in_executor(
                None, lambda: model.generate(prompts=[prompt], sampling_params=sampling)
            )
            output = outputs[0]
            text = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            
            # Add basic metrics to span
            duration = time.time() - start_time
            span.set_attribute("duration_ms", int(duration * 1000))
            span.set_attribute("num_tokens", num_tokens)
            span.set_attribute("max_tokens", req.max_tokens)
            span.set_attribute("message_count", len(req.messages))
            
            return ChatResponse(text=text, num_tokens=num_tokens)
        except Exception as e:
            span.set_attribute("error", True)
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False) 