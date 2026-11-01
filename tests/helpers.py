"""Shared pieces for the offline tests: the config, fake chat completions, a mock HTTP client and sample rows."""
import json

import httpx
import yaml

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))


def completion(content, reasoning=None, finish="stop", reasoning_tokens=0, completion_tokens=50, logprobs=None):
    """A vLLM 0.32 chat completion: the reasoning parser puts the thinking in `reasoning`."""
    choice = {"finish_reason": finish, "message": {"content": content, "reasoning": reasoning}}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"choices": [choice], "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens,
                                           "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}}


def mock_client(handler, bodies=None):
    """httpx client answering every request with handler(path, json_body); chat bodies are appended to `bodies`."""
    def respond(request):
        body = json.loads(request.content) if request.content else None
        if bodies is not None and request.url.path.endswith("/chat/completions"):
            bodies.append(body)
        return httpx.Response(200, json=handler(request.url.path, body))
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond))


def gen_row(i, **kw):
    """A generated sample as it reaches filter/judge/select."""
    base = {"id": f"p{i}", "sample_idx": 0, "source_dataset": "no_robots", "input": f"Question {i}?",
            "reasoning": "Long trace.", "rationale": "", "answer": "Answer.", "finish_reason": "stop",
            "think_unclosed": False, "seed": 5, "mode": "open", "thinking": True, "cot_method": None, "closed": False}
    return {**base, **kw}


# Source questions of a Thai set: p0-p3 were used in English, so translation picks from the other eight.
SOURCE = [{"id": f"p{i}", "input": f"Name 3 fruits high in vitamin C, take {'abcdefghijkl'[i]}.", "strata": f"s{i % 2}",
           "closed": False, "reference": f"Reference {i}.", "hint_ok": i % 2 == 0, "source_dataset": "no_robots",
           "gold": []} for i in range(12)]
THAI = "บอกชื่อผลไม้ 3 ชนิดที่มีวิตามินซีสูง"
