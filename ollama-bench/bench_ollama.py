import json, time, csv, statistics, subprocess, argparse
from datetime import datetime

TEMP = 0.0          # same temperature for all runs
NUM_CTX = 2048      # keep constant across models
LOOPS = 5           # measured loops (warm-up is extra)

# Output caps (fixed/limited output length)
OUT_LARGE = 512     # small input, large output
OUT_SMALL = 128      # large input, small output

# Prompt files
SMALL_PROMPT_FILE = "prompt_small.txt"
LARGE_PROMPT_FILE = "prompt_large.txt"

# =============================

def call_generate(model: str, prompt: str, num_predict: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMP,
            "num_predict": num_predict,
            "num_ctx": NUM_CTX,
        }
    }
    # Use curl to local Ollama API
    cmd = ["curl", "-s", "http://localhost:11434/api/generate", "-d", json.dumps(payload)]
    out = subprocess.check_output(cmd)
    return json.loads(out)

def ns_to_s(x): return x / 1e9

def safe_rate(tok, dur_ns):
    if not dur_ns or dur_ns <= 0:
        return 0.0
    return tok / ns_to_s(dur_ns)

def run_scenario(scenario_name: str, model: str, prompt_text: str, num_predict: int, writer):
    print(f"\n=== Scenario: {scenario_name} | model={model} | num_predict={num_predict} | temp={TEMP} | num_ctx={NUM_CTX} ===")

    # Warm-up (not counted)
    _ = call_generate(model, "hi", 8)

    decode_rates = []
    for t in range(1, LOOPS + 1):
        r = call_generate(model, prompt_text, num_predict)

        total_s = ns_to_s(r.get("total_duration", 0))
        load_s  = ns_to_s(r.get("load_duration", 0))
        p_tok   = r.get("prompt_eval_count", 0)
        p_ns    = r.get("prompt_eval_duration", 0)
        g_tok   = r.get("eval_count", 0)
        g_ns    = r.get("eval_duration", 0)

        p_rate = safe_rate(p_tok, p_ns)
        g_rate = safe_rate(g_tok, g_ns)

        decode_rates.append(g_rate)

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": scenario_name,
            "model": model,
            "trial": t,
            "temperature": TEMP,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
            "prompt_tokens": p_tok,
            "prefill_s": ns_to_s(p_ns),
            "prefill_tok_s": p_rate,
            "gen_tokens": g_tok,
            "decode_s": ns_to_s(g_ns),
            "decode_tok_s": g_rate,
            "load_s": load_s,
            "total_s": total_s,
        })

        print(f"{model:35s} trial {t}: prefill {p_rate:8.2f} tok/s | decode {g_rate:8.2f} tok/s | total {total_s:6.2f}s | load {load_s:5.2f}s")

        # 5 sec gap between trials (avoid GPU hang / cool down)
        if t != LOOPS:
            time.sleep(5)

    med = statistics.median(decode_rates) if decode_rates else 0.0
    print(f"--> {model}: median decode tok/s over {LOOPS} trials = {med:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to benchmark (single model only)")
    parser.add_argument(
        "--scenario",
        choices=["small_in_large_out", "large_in_small_out"],
        default="small_in_large_out",
        help="Which scenario to run (default: small_in_large_out)"
    )
    args = parser.parse_args()

    model = args.model
    scenario = args.scenario

    model_safe = model.replace(":", "_").replace("/", "_")
    out_csv = f"ollama_bench_{model_safe}_{scenario}_{int(time.time())}.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp","scenario","model","trial","temperature","num_ctx","num_predict",
            "prompt_tokens","prefill_s","prefill_tok_s","gen_tokens","decode_s","decode_tok_s",
            "load_s","total_s"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if scenario == "small_in_large_out":
            prompt_text = open(SMALL_PROMPT_FILE, "r", encoding="utf-8").read().strip()
            run_scenario("small_in_large_out", model, prompt_text, OUT_LARGE, writer)
        else:  # large_in_small_out
            prompt_text = open(LARGE_PROMPT_FILE, "r", encoding="utf-8").read().strip()
            run_scenario("large_in_small_out", model, prompt_text, OUT_SMALL, writer)

    print(f"\nSaved CSV: {out_csv}")



if __name__ == "__main__":
    main()

