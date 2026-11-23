"""Capture a llama.cpp reference for the WebGPU parity test (scripts/test-model-parity.mts).

For each prompt: the prompt tokens, 24 greedy tokens and the full-vocab top-5
probabilities at every step, from a running llama-server. Run it once against
a CPU server and once against a GPU server (--cuda-url) so the fixture also
holds llama.cpp's own CPU-vs-GPU spread, which the test uses as its tolerance.

    llama-server -m MODEL.gguf -c 4096 -ngl 0 --jinja --port 8090      (CPU)
    llama-server -m MODEL.gguf -c 4096 -ngl 99 --jinja --port 8091     (GPU)
    python scripts/gen-llamacpp-reference.py --url http://127.0.0.1:8090 \
        --cuda-url http://127.0.0.1:8091 --model-desc "repo@rev file" --out test-fixtures/x.json

Standard library only.
"""
import argparse, json, math, urllib.request

LONG = ("The river ran past the old mill, where the miller kept a ledger of every sack of grain "
        "that came through his door. ") * 1
STEPS = 24
CHAT = [{"role": "user", "content": "Name three colours of the rainbow, one per line."}]


def post(url, path, body):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def long_prompt():
    # ~700 tokens of varied prose (crosses a 512-token prefill chunk)
    parts = []
    for i in range(36):
        parts.append(f"In year {1800 + i * 3}, the ledger listed {7 + i * 11} sacks of wheat, "
                     f"{3 + i * 5} of barley and a note about the weather on day {i + 1}.")
    return LONG + " ".join(parts) + " The miller closed the ledger and said:"


def capture(url, tokens):
    out = post(url, "/completion", {"prompt": tokens, "n_predict": STEPS, "temperature": 0, "top_k": 1,
                                    "n_probs": 5, "cache_prompt": False, "samplers": ["top_k"]})
    steps = []
    for p in out["completion_probabilities"]:
        top = [{"id": t["id"], "prob": math.exp(t["logprob"])} for t in p["top_logprobs"]]
        steps.append({"token": p["id"], "top": top})
    return steps, out.get("content", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--cuda-url", required=True)
    ap.add_argument("--model-desc", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--source-cuda", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chat_prompt = post(a.url, "/apply-template", {
        "messages": CHAT, "chat_template_kwargs": {"enable_thinking": True}})["prompt"]
    cases = {}
    for name, text in (("short", "The capital of France is"), ("long", long_prompt()),
                       ("chat_template", chat_prompt)):
        toks = post(a.url, "/tokenize", {"content": text, "add_special": name != "chat_template",
                                         "parse_special": True})["tokens"]
        steps, content = capture(a.url, toks)
        steps_cuda, _ = capture(a.cuda_url, toks)
        gen = [s["token"] for s in steps]
        if [s["token"] for s in steps_cuda] != gen:
            print(f"note: {name}: GPU greedy differs from CPU (kept; the test compares probabilities)")
        cases[name] = {"prompt": text, "n_prompt_tokens": len(toks), "prompt_tokens": toks,
                       "generated_tokens": gen, "generated_text": content,
                       "steps": steps, "steps_cuda": steps_cuda}
        print(f"{name}: {len(toks)} prompt tokens -> {content[:60]!r}")
    json.dump({"source": a.source, "source_cuda": a.source_cuda, "model": a.model_desc,
               "captured_with": "llama-server /tokenize (add_special except the rendered chat template, "
                                "parse_special) + /completion temperature 0 top_k 1 n_probs 5 "
                                "cache_prompt false; prob = exp(top_logprobs.logprob); chat template "
                                "from /apply-template with enable_thinking true",
               "chat_messages": CHAT, "cases": cases},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
