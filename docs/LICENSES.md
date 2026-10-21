# Licenses

Licenses were checked against the `license:` field of each Hugging Face card on 2026-10-13.
We use only the prompt text as model input. Original responses and ratings are used only to pick hint references and to spot-check outputs. They never appear as output.

| Item | License | Source | Allowed for our use? | Notes |
|---|---|---|---|---|
| HuggingFaceH4/no_robots | CC BY-NC 4.0 | https://huggingface.co/datasets/HuggingFaceH4/no_robots | Yes, **non-commercial only** (accepted by the group) | **Everything derived from No Robots inherits the non-commercial restriction**: generated outputs, filtered sets, samples and any model trained on them. Keep its rows identifiable via `source_dataset`. Attribution required. |
| nvidia/HelpSteer2 | CC BY 4.0 | https://huggingface.co/datasets/nvidia/HelpSteer2 | Yes | Attribution required. |
| openbmb/UltraFeedback | MIT | https://huggingface.co/datasets/openbmb/UltraFeedback | Yes | Keep the copyright and license notice. Its prompts come from upstream sources (ShareGPT, FLAN, Evol-Instruct, UltraChat, TruthfulQA, FalseQA), which may have their own terms. TODO: a human should review the upstream sources. |
| Teacher: Qwen/Qwen3.5-9B | Apache-2.0 (per HF card) | https://huggingface.co/Qwen/Qwen3.5-9B | **TODO: human to check** | Confirm that using the outputs to train another model is permitted. Run locally with vLLM from 2026-10-18. |
| Judge: google/gemma-4-31B-it-qat-w4a16-ct (local) and google/gemma-4-31b-it (OpenRouter) | Apache-2.0 (per HF card, checked 2026-10-18) | https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct | Yes | Only its scores and critiques are used, to select samples. Its outputs are never training targets. |
| Student: wongwian-org/wongwian-micro-instruct | MIT (per HF card, checked 2026-10-18) | https://huggingface.co/wongwian-org/wongwian-micro-instruct | Yes | Only its tokenizer is used here, to check that samples fit its 2,048-token context. |
| Reasoning Complexity Rating Prompt | Quoted from Hu et al. (2026), "DeReason", arXiv:2603.11193, Appendix A.1 | https://arxiv.org/abs/2603.11193 | Yes, with citation | Reproduced verbatim in `prompts/rate_v1.txt` and cited in `distill/rate.py` and the README. |
| HelpSteer2 validation split (judge validation) | CC BY 4.0 | https://huggingface.co/datasets/nvidia/HelpSteer2 | Yes | Used only to measure judge agreement with human ratings; never generated from. |
| OpenRouter terms of service | — | https://openrouter.ai/terms | **TODO: human to check** | Confirm the rules on using outputs for training. Applies only to data generated through OpenRouter: the 2026-10-13 pilot and laptop smoke tests. |
| Upstream inference providers (DeepInfra, Parasail, SiliconFlow, Venice, Together; see the `provider` field in each row) | — | provider websites | **TODO: human to check** | Each provider that serves the model may have its own terms for outputs. Rows generated locally have `provider: local-vllm`. |
| Compute: company GPU VM | — | — | **TODO: human to check** | No Robots is non-commercial. Confirm that running it on company hardware doesn't make the use commercial. |
