"""Shared pieces for the offline tests: the config, fake chat completions and a mock HTTP client."""
import json

import httpx
import yaml

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))


def completion(content, reasoning=None, finish="stop", reasoning_tokens=0, completion_tokens=50, logprobs=None):
    choice = {"finish_reason": finish, "message": {"content": content, "reasoning": reasoning}}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"provider": "Test", "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens, "cost": 0.001,
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
