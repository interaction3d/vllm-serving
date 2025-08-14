import os
import asyncio
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

load_dotenv()

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


async def get_llm() -> LLM:
    global _llm
    if _llm is None:
        def _init_llm() -> LLM:
            return LLM(
                model=MODEL_NAME,
                tensor_parallel_size=TP_DEGREE,
                trust_remote_code=TRUST_REMOTE_CODE,
                max_model_len=MAX_MODEL_LEN,
            )

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
    try:
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
            None, lambda: model.generate(prompts=[req.prompt], sampling_params=sampling)
        )
        output = outputs[0]
        text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        return GenerateResponse(text=text, num_tokens=num_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
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
        return ChatResponse(text=text, num_tokens=num_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False) 