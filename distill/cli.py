"""Command line entry point: distill prepare | subsets | serve | wait | rate | generate | compress | filter | judge |
select | analyze | push."""
import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

from distill import compress, data, generate, hub, judge, rate, report, selection, server

BACKENDS = ["vllm", "openrouter"]
ONLY_HELP = "only prompts with (closed) or without (open) a known short answer"


def main(argv=None):
    try:
        run(argv)
    except KeyboardInterrupt:
        # Records are flushed one by one. Exit without joining worker threads still blocked on a reply: a normal exit
        # waits for them, and on Ctrl-C the vLLM server stops answering while it waits for these same connections.
        print("interrupted", file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(130)
    except server.ServerDown as e:  # same quick exit; pipeline.sh restarts the server on this code and resumes
        print(f"server down: {e}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(server.SERVER_DOWN_EXIT)


def run(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    # Machine settings (HF_HOME and the like), also for commands typed without `source .vm_env`: without them,
    # downloads land on the VM's nearly full root disk. Variables already set win; the Hugging Face libraries are
    # imported lazily, so they still read these.
    load_dotenv(".vm_env")
    ap = argparse.ArgumentParser(prog="distill")
    ap.add_argument("--config", default=os.environ.get("DISTILL_CONFIG", "config.yaml"),
                    help="default: $DISTILL_CONFIG or config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="load, dedupe and cache prompts from all datasets").add_argument(
        "--dataset", help="build only this dataset (one added at the end), deduped against the earlier ones")
    sub.add_parser("subsets", help="print which sets a dataset has: open and/or closed").add_argument("dataset")

    sub.add_parser("serve", help="run `vllm serve` for the teacher or judge (GPU VM)").add_argument(
        "role", choices=["teacher", "judge"])
    w = sub.add_parser("wait", help="block until the local vLLM server for a role is ready")
    w.add_argument("role", choices=["teacher", "judge"])
    w.add_argument("--pid", type=int, help="server process to watch: fail at once if it exits")

    r = sub.add_parser("rate", help="rate each question's reasoning complexity 1-5 (DeReason prompt)")
    r.add_argument("--dataset", required=True)
    r.add_argument("--only", choices=["closed", "open"], help=ONLY_HELP)
    r.add_argument("--limit", type=int, help="same seeded stratified sample as generate with the same flags")
    r.add_argument("--seed", type=int)
    r.add_argument("--concurrency", type=int)
    r.add_argument("--backend", choices=BACKENDS)

    g = sub.add_parser("generate", help="query the teacher N times per prompt and append JSONL records")
    g.add_argument("--dataset", required=True)
    g.add_argument("--model")
    g.add_argument("--mode", choices=["open", "hint", "cot"], default="open",
                   help="cot: thinking off, short <reasoning> section before the answer")
    g.add_argument("--thinking", choices=["on", "off"], default="on")
    g.add_argument("--n", type=int, help="samples per prompt")
    g.add_argument("--only", choices=["closed", "open"], help=ONLY_HELP)
    g.add_argument("--limit", type=int, help="pilot: seeded stratified sample of N prompts")
    g.add_argument("--seed", type=int)
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--concurrency", type=int)
    g.add_argument("--max-cost", type=float, help="stop dispatching once this many USD are spent")
    g.add_argument("--prompt-version")
    g.add_argument("--include-reference", action="store_true")
    g.add_argument("--backend", choices=BACKENDS)
    g.add_argument("--out", required=True)

    c = sub.add_parser("compress", help="rewrite thinking-on traces into short rationales (variant B)")
    c.add_argument("file")
    c.add_argument("--out", required=True)
    c.add_argument("--concurrency", type=int)
    c.add_argument("--backend", choices=BACKENDS)

    a = sub.add_parser("analyze", help="write a Markdown report comparing variants, scores and lengths")
    a.add_argument("files", nargs="+")
    a.add_argument("--out", default="reports/report.md")

    f = sub.add_parser("filter", help="drop rows that fail heuristic checks")
    f.add_argument("file")
    f.add_argument("--out", required=True)
    f.add_argument("--max-reasoning-tokens", type=int)
    f.add_argument("--max-student-tokens", type=int, help="student context budget (default from config; 0 disables)")

    j = sub.add_parser("judge", help="score rationale + answer of filtered samples with the judge model")
    j.add_argument("files", nargs="*")
    j.add_argument("--validate", type=int, metavar="N", help="first measure agreement with humans on N HelpSteer2 pairs")
    j.add_argument("--all", action="store_true",
                   help="judge every sample, including rule-failing closed ones (to measure judge vs rules)")
    j.add_argument("--thinking", choices=["on", "off"], help="override judge.thinking from config")
    j.add_argument("--prompt-version", help="override judge.prompt_version (rubric) from config")
    j.add_argument("--concurrency", type=int)
    j.add_argument("--backend", choices=BACKENDS)

    s = sub.add_parser("select", help="keep the best samples and write them in curriculum order")
    s.add_argument("file")
    s.add_argument("--out", required=True)
    s.add_argument("--order", choices=["curriculum", "random"])
    s.add_argument("--min-score", type=int)

    p = sub.add_parser("push", help="back up a run and the ratings to a private Hugging Face dataset repo")
    p.add_argument("out", help="the run's output directory, e.g. data/full")
    p.add_argument("--repo", help="dataset repo id (default: config hub.repo)")

    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.cmd == "prepare":
        data.prepare(cfg, only=args.dataset)
    elif args.cmd == "subsets":
        print(" ".join(data.subsets(cfg, args.dataset)))
    elif args.cmd == "serve":
        server.serve(cfg, args.role)
    elif args.cmd == "wait":
        server.wait_ready(cfg, args.role, pid=args.pid)
    elif args.cmd == "rate":
        rate.run(cfg, args.dataset, only=args.only, limit=args.limit, seed=args.seed, concurrency=args.concurrency,
                 backend=args.backend)
    elif args.cmd == "generate":
        generate.run(cfg, args.dataset, mode=args.mode, thinking=args.thinking == "on", limit=args.limit,
                     seed=args.seed, max_tokens=args.max_tokens, out=args.out, n=args.n, concurrency=args.concurrency,
                     max_cost=args.max_cost, include_reference=args.include_reference, model=args.model,
                     prompt_version=args.prompt_version, only=args.only, backend=args.backend)
    elif args.cmd == "compress":
        compress.run(cfg, args.file, args.out, concurrency=args.concurrency, backend=args.backend)
    elif args.cmd == "analyze":
        report.analyze(args.files, args.out, min_score=cfg["select"]["min_score"], student=cfg["student"]["tokenizer"])
    elif args.cmd == "filter":
        budget = cfg["student"]["max_total_tokens"] if args.max_student_tokens is None else args.max_student_tokens
        report.filter_file(args.file, args.out, cfg["filter"], args.max_reasoning_tokens, cfg["cot"]["max_words"],
                           (cfg["student"]["tokenizer"], budget) if budget else None)
    elif args.cmd == "judge":
        if not (args.files or args.validate):
            ap.error("judge needs FILES and/or --validate N")
        thinking = None if args.thinking is None else args.thinking == "on"
        if args.validate:
            judge.validate(cfg, args.validate, concurrency=args.concurrency, backend=args.backend, thinking=thinking,
                           prompt_version=args.prompt_version)
        if args.files:
            judge.run(cfg, args.files, concurrency=args.concurrency, backend=args.backend, all_samples=args.all,
                      thinking=thinking, prompt_version=args.prompt_version)
    elif args.cmd == "select":
        selection.run(cfg, args.file, args.out, order=args.order, min_score=args.min_score)
    elif args.cmd == "push":
        hub.push(cfg, args.out, repo=args.repo)


if __name__ == "__main__":
    main()
