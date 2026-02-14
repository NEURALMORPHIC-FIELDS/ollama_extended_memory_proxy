#!/usr/bin/env python3
"""
================================================================================
Ollama Memory Proxy - Comprehensive Benchmark Suite
================================================================================

Compares LLM performance WITH vs WITHOUT the memory proxy across:
  1. Personal fact recall (name, location, job, preferences)
  2. Multi-turn conversation continuity
  3. Temporal context (tracking evolving information)
  4. Cross-topic association (connecting facts from different conversations)
  5. Latency overhead measurement

Each test category seeds facts via setup messages, then queries the LLM
in a fresh conversation (no chat history) to isolate memory contribution.

Author: Benchmark for ollama-memory-proxy
================================================================================
"""

import json
import time
import sys
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "qwen3:0.6b"
DIRECT_URL = "http://127.0.0.1:11434"   # Ollama direct (no memory)
PROXY_URL = "http://127.0.0.1:11435"    # Memory proxy
TIMEOUT = 600.0  # seconds per request (first call may cold-load model)
SETTLING_DELAY = 2.0  # seconds to wait after seeding for background storage


@dataclass
class TestResult:
    test_id: str
    category: str
    query: str
    expected_keywords: List[str]
    response: str
    keywords_found: List[str]
    keywords_missing: List[str]
    score: float  # 0.0 - 1.0
    latency_ms: float
    pass_fail: str  # "PASS" / "PARTIAL" / "FAIL"


@dataclass
class BenchmarkRun:
    mode: str  # "direct" or "proxy"
    url: str
    model: str
    results: List[TestResult] = field(default_factory=list)
    seed_latencies_ms: List[float] = field(default_factory=list)
    total_time_s: float = 0.0


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def chat(base_url: str, messages: List[Dict], model: str = MODEL) -> Tuple[str, float]:
    """Send a chat request, return (response_text, latency_ms)."""
    client = httpx.Client(base_url=base_url, timeout=TIMEOUT)
    t0 = time.perf_counter()
    r = client.post("/api/chat", json={
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    })
    latency = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    text = data.get("message", {}).get("content", "")
    # Strip <think>...</think> blocks from qwen3 models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    client.close()
    return text, latency


def seed_fact(base_url: str, message: str, model: str = MODEL) -> float:
    """Send a fact-establishing message, return latency_ms."""
    _, latency = chat(base_url, [
        {"role": "user", "content": message}
    ], model)
    return latency


def query_fresh(base_url: str, question: str, model: str = MODEL) -> Tuple[str, float]:
    """Ask a question in a brand-new conversation (no history), return (response, latency_ms)."""
    return chat(base_url, [
        {"role": "user", "content": question}
    ], model)


# ---------------------------------------------------------------------------
# Keyword-based scoring
# ---------------------------------------------------------------------------

def score_response(response: str, expected_keywords: List[str]) -> Tuple[List[str], List[str], float]:
    """Check which expected keywords appear in the response (case-insensitive)."""
    lower = response.lower()
    found = [kw for kw in expected_keywords if kw.lower() in lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in lower]
    score = len(found) / len(expected_keywords) if expected_keywords else 0.0
    return found, missing, score


def pass_fail(score: float) -> str:
    if score >= 0.8:
        return "PASS"
    elif score >= 0.4:
        return "PARTIAL"
    else:
        return "FAIL"


# ---------------------------------------------------------------------------
# Test Definitions
# ---------------------------------------------------------------------------

# Each test: (category, seed_messages, query, expected_keywords, test_id)

TESTS = [
    # ===== CATEGORY 1: Personal Fact Recall =====
    {
        "category": "1. Personal Fact Recall",
        "seeds": [
            "My name is Lucian Borbeleac and I am 35 years old.",
        ],
        "query": "What is my name and how old am I?",
        "expected": ["Lucian", "35"],
        "id": "PF-01-name-age",
    },
    {
        "category": "1. Personal Fact Recall",
        "seeds": [
            "I live in Timisoara, Romania and I work as an AI researcher.",
        ],
        "query": "Where do I live and what is my profession?",
        "expected": ["Timisoara", "Romania", "AI"],
        "id": "PF-02-location-job",
    },
    {
        "category": "1. Personal Fact Recall",
        "seeds": [
            "My favorite programming language is Python and I use Neovim as my editor.",
        ],
        "query": "What programming language do I prefer and what editor do I use?",
        "expected": ["Python", "Neovim"],
        "id": "PF-03-preferences",
    },
    {
        "category": "1. Personal Fact Recall",
        "seeds": [
            "I have a black cat named Shadow and a golden retriever named Rex.",
        ],
        "query": "What pets do I have? What are their names?",
        "expected": ["Shadow", "Rex", "cat", "retriever"],
        "id": "PF-04-pets",
    },
    {
        "category": "1. Personal Fact Recall",
        "seeds": [
            "My phone number is 0722-555-123 and my email is lucian@example.com.",
        ],
        "query": "What is my phone number and email address?",
        "expected": ["0722", "555", "123", "lucian@example.com"],
        "id": "PF-05-contact",
    },

    # ===== CATEGORY 2: Multi-Turn Conversation Continuity =====
    {
        "category": "2. Multi-Turn Continuity",
        "seeds": [
            "I am building a drone that can deliver packages autonomously.",
            "The drone uses a Raspberry Pi 5 as its flight controller.",
            "I plan to add computer vision using a stereo camera setup.",
        ],
        "query": "What project am I working on and what hardware does it use?",
        "expected": ["drone", "Raspberry Pi", "camera"],
        "id": "MT-01-project-hw",
    },
    {
        "category": "2. Multi-Turn Continuity",
        "seeds": [
            "Yesterday I went to the dentist for a root canal procedure.",
            "The dentist said I need to come back next week for a crown fitting.",
            "The total cost will be around 500 euros.",
        ],
        "query": "What medical appointment did I have recently and what follow-up is needed?",
        "expected": ["dentist", "root canal", "crown", "500"],
        "id": "MT-02-medical",
    },
    {
        "category": "2. Multi-Turn Continuity",
        "seeds": [
            "I am reading the book 'Godel Escher Bach' by Douglas Hofstadter.",
            "I am on chapter 12 and I find the recursion concepts fascinating.",
            "After this I plan to read 'The Master Algorithm' by Pedro Domingos.",
        ],
        "query": "What book am I currently reading and what will I read next?",
        "expected": ["Godel", "Hofstadter", "Master Algorithm", "Domingos"],
        "id": "MT-03-books",
    },

    # ===== CATEGORY 3: Temporal Context (evolving info) =====
    {
        "category": "3. Temporal Context",
        "seeds": [
            "I just bought a new laptop - a ThinkPad X1 Carbon Gen 12 with 32GB RAM.",
            "I installed Ubuntu 24.04 on it and it runs perfectly.",
            "Update: I upgraded the RAM to 64GB because I need it for ML training.",
        ],
        "query": "What laptop do I have and how much RAM does it have now?",
        "expected": ["ThinkPad", "64"],
        "id": "TC-01-laptop-update",
    },
    {
        "category": "3. Temporal Context",
        "seeds": [
            "I moved to a new apartment on Strada Victoriei, 3rd floor.",
            "The rent is 450 euros per month, utilities included.",
            "Update: the landlord raised the rent to 500 euros starting next month.",
        ],
        "query": "Where is my apartment and how much is my rent?",
        "expected": ["Victoriei", "500"],
        "id": "TC-02-rent-update",
    },

    # ===== CATEGORY 4: Cross-Topic Association =====
    {
        "category": "4. Cross-Topic Association",
        "seeds": [
            "My sister's name is Elena and she lives in Cluj-Napoca.",
            "Elena works as a doctor at the county hospital.",
            "I need to visit Elena next month because she is getting married on March 15th.",
        ],
        "query": "What does my sister do for work, where does she live, and when is she getting married?",
        "expected": ["Elena", "doctor", "Cluj", "March", "15"],
        "id": "CT-01-sister",
    },
    {
        "category": "4. Cross-Topic Association",
        "seeds": [
            "My car is a 2022 Dacia Duster with 45000 km on it.",
            "The car needs new brake pads - the mechanic quoted 200 euros.",
            "I also need to renew my insurance before April 1st, the premium is 800 euros per year.",
        ],
        "query": "Tell me about my car - what model is it, what repairs does it need, and when is the insurance due?",
        "expected": ["Dacia", "Duster", "brake", "200", "April", "insurance"],
        "id": "CT-02-car",
    },

    # ===== CATEGORY 5: Precision & Detail Recall =====
    {
        "category": "5. Precision Recall",
        "seeds": [
            "The WiFi password for my home network is Tr0ub4dor&3. The network name is BorbeNet5G.",
        ],
        "query": "What is my WiFi network name and password?",
        "expected": ["BorbeNet5G", "Tr0ub4dor"],
        "id": "PR-01-wifi",
    },
    {
        "category": "5. Precision Recall",
        "seeds": [
            "My server's IP address is 192.168.1.42 and it runs on port 8080.",
            "The admin username is 'lucian_admin' and the database is PostgreSQL 16.",
        ],
        "query": "What is my server IP, port, admin username, and database engine?",
        "expected": ["192.168.1.42", "8080", "lucian_admin", "PostgreSQL"],
        "id": "PR-02-server",
    },
]


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(mode: str, base_url: str, model: str = MODEL) -> BenchmarkRun:
    """Run all tests against a given endpoint."""
    run = BenchmarkRun(mode=mode, url=base_url, model=model)
    t_start = time.perf_counter()

    total = len(TESTS)
    for i, test in enumerate(TESTS, 1):
        tid = test["id"]
        category = test["category"]
        print(f"  [{i}/{total}] {tid} ...", end=" ", flush=True)

        # Seed facts
        for seed_msg in test["seeds"]:
            lat = seed_fact(base_url, seed_msg, model)
            run.seed_latencies_ms.append(lat)

        # Wait for background storage (proxy stores async)
        if mode == "proxy":
            time.sleep(SETTLING_DELAY)

        # Query in fresh conversation
        response, query_lat = query_fresh(base_url, test["query"], model)
        found, missing, sc = score_response(response, test["expected"])
        pf = pass_fail(sc)

        result = TestResult(
            test_id=tid,
            category=category,
            query=test["query"],
            expected_keywords=test["expected"],
            response=response[:500],
            keywords_found=found,
            keywords_missing=missing,
            score=sc,
            latency_ms=query_lat,
            pass_fail=pf,
        )
        run.results.append(result)
        print(f"{pf} (score={sc:.0%}, {query_lat:.0f}ms)")

    run.total_time_s = time.perf_counter() - t_start
    return run


# ---------------------------------------------------------------------------
# Latency Benchmark (pure overhead measurement)
# ---------------------------------------------------------------------------

def run_latency_benchmark(base_url: str, mode: str, n_rounds: int = 5) -> Dict[str, Any]:
    """Measure pure latency overhead with simple queries."""
    print(f"\n  Latency benchmark ({n_rounds} rounds)...")
    latencies = []
    for i in range(n_rounds):
        msg = f"Say only the word 'hello'. Nothing else. Round {i+1}."
        _, lat = query_fresh(base_url, msg)
        latencies.append(lat)
        print(f"    Round {i+1}: {lat:.0f}ms")

    return {
        "mode": mode,
        "rounds": n_rounds,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "raw_ms": latencies,
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    direct: BenchmarkRun,
    proxy: BenchmarkRun,
    lat_direct: Dict,
    lat_proxy: Dict,
    output_path: str,
):
    """Generate a markdown benchmark report."""
    lines = []
    L = lines.append

    L("# Ollama Memory Proxy - Benchmark Report")
    L(f"")
    L(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"**Model:** {MODEL}")
    L(f"**Direct Ollama:** {DIRECT_URL}")
    L(f"**Memory Proxy:** {PROXY_URL}")
    L(f"**Total tests:** {len(TESTS)}")
    L("")

    # ---- Executive Summary ----
    d_scores = [r.score for r in direct.results]
    p_scores = [r.score for r in proxy.results]
    d_pass = sum(1 for r in direct.results if r.pass_fail == "PASS")
    p_pass = sum(1 for r in proxy.results if r.pass_fail == "PASS")
    d_fail = sum(1 for r in direct.results if r.pass_fail == "FAIL")
    p_fail = sum(1 for r in proxy.results if r.pass_fail == "FAIL")

    d_avg = statistics.mean(d_scores) * 100
    p_avg = statistics.mean(p_scores) * 100
    improvement = p_avg - d_avg

    L("## Executive Summary")
    L("")
    L("| Metric | Direct (no memory) | Memory Proxy | Delta |")
    L("|--------|-------------------|-------------|-------|")
    L(f"| **Average Score** | {d_avg:.1f}% | {p_avg:.1f}% | **+{improvement:.1f}%** |")
    L(f"| Tests PASSED | {d_pass}/{len(TESTS)} | {p_pass}/{len(TESTS)} | +{p_pass - d_pass} |")
    L(f"| Tests FAILED | {d_fail}/{len(TESTS)} | {p_fail}/{len(TESTS)} | {p_fail - d_fail:+d} |")
    L(f"| Mean Query Latency | {lat_direct['mean_ms']:.0f}ms | {lat_proxy['mean_ms']:.0f}ms | +{lat_proxy['mean_ms'] - lat_direct['mean_ms']:.0f}ms |")
    L(f"| Total Runtime | {direct.total_time_s:.1f}s | {proxy.total_time_s:.1f}s | |")
    L("")

    # ---- Per-Category Breakdown ----
    L("## Per-Category Breakdown")
    L("")
    categories = sorted(set(t["category"] for t in TESTS))
    L("| Category | Direct Avg | Proxy Avg | Improvement |")
    L("|----------|-----------|----------|-------------|")
    for cat in categories:
        d_cat = [r.score for r in direct.results if r.category == cat]
        p_cat = [r.score for r in proxy.results if r.category == cat]
        d_a = statistics.mean(d_cat) * 100 if d_cat else 0
        p_a = statistics.mean(p_cat) * 100 if p_cat else 0
        L(f"| {cat} | {d_a:.0f}% | {p_a:.0f}% | **+{p_a - d_a:.0f}%** |")
    L("")

    # ---- Detailed Test Results ----
    L("## Detailed Test Results")
    L("")
    L("| Test ID | Category | Direct | Proxy | Keywords Found (Proxy) |")
    L("|---------|----------|--------|-------|----------------------|")
    for dr, pr in zip(direct.results, proxy.results):
        d_icon = {"PASS": "PASS", "PARTIAL": "PART", "FAIL": "FAIL"}[dr.pass_fail]
        p_icon = {"PASS": "PASS", "PARTIAL": "PART", "FAIL": "FAIL"}[pr.pass_fail]
        kw = ", ".join(pr.keywords_found) if pr.keywords_found else "-"
        L(f"| {dr.test_id} | {dr.category.split('.')[0]}. | {d_icon} ({dr.score:.0%}) | {p_icon} ({pr.score:.0%}) | {kw} |")
    L("")

    # ---- Latency Analysis ----
    L("## Latency Analysis")
    L("")
    L("| Metric | Direct | Proxy | Overhead |")
    L("|--------|--------|-------|----------|")
    L(f"| Mean | {lat_direct['mean_ms']:.0f}ms | {lat_proxy['mean_ms']:.0f}ms | +{lat_proxy['mean_ms'] - lat_direct['mean_ms']:.0f}ms |")
    L(f"| Median | {lat_direct['median_ms']:.0f}ms | {lat_proxy['median_ms']:.0f}ms | +{lat_proxy['median_ms'] - lat_direct['median_ms']:.0f}ms |")
    L(f"| Min | {lat_direct['min_ms']:.0f}ms | {lat_proxy['min_ms']:.0f}ms | |")
    L(f"| Max | {lat_direct['max_ms']:.0f}ms | {lat_proxy['max_ms']:.0f}ms | |")
    if lat_direct['mean_ms'] > 0:
        overhead_pct = ((lat_proxy['mean_ms'] - lat_direct['mean_ms']) / lat_direct['mean_ms']) * 100
        L(f"| **Overhead %** | - | - | **{overhead_pct:+.1f}%** |")
    L("")

    # ---- Sample Responses ----
    L("## Sample Responses (Selected Tests)")
    L("")
    # Pick a few interesting test IDs
    samples = ["PF-01-name-age", "MT-01-project-hw", "CT-01-sister", "PR-02-server"]
    for sid in samples:
        dr = next((r for r in direct.results if r.test_id == sid), None)
        pr = next((r for r in proxy.results if r.test_id == sid), None)
        if not dr or not pr:
            continue
        L(f"### {sid}: {dr.query}")
        L(f"")
        L(f"**Expected keywords:** {', '.join(dr.expected_keywords)}")
        L(f"")
        L(f"**Direct (score: {dr.score:.0%}):**")
        L(f"> {dr.response[:300]}")
        L(f"")
        L(f"**Proxy (score: {pr.score:.0%}):**")
        L(f"> {pr.response[:300]}")
        L(f"")

    # ---- Methodology ----
    L("## Methodology")
    L("")
    L("1. **Seed phase:** Facts are sent as individual chat messages to establish memory context")
    L("2. **Query phase:** Questions are asked in a **brand-new conversation** (no chat history)")
    L("3. **Scoring:** Keyword presence check (case-insensitive) against expected facts")
    L("4. **Pass criteria:** >=80% keywords found = PASS, >=40% = PARTIAL, <40% = FAIL")
    L("5. **Direct mode:** Queries go straight to Ollama (no memory) - LLM has no prior context")
    L("6. **Proxy mode:** Queries go through memory proxy which injects relevant past conversations")
    L("7. **Latency:** Measured separately with minimal queries to isolate proxy overhead")
    L("8. **Temperature:** 0.1 for reproducibility")
    L("")

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  OLLAMA MEMORY PROXY - COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    print(f"  Model:        {MODEL}")
    print(f"  Direct:       {DIRECT_URL}")
    print(f"  Proxy:        {PROXY_URL}")
    print(f"  Tests:        {len(TESTS)}")
    print("=" * 70)

    # Verify connectivity
    print("\nVerifying connectivity...")
    try:
        httpx.get(f"{DIRECT_URL}/api/tags", timeout=5).raise_for_status()
        print(f"  Ollama (direct): OK")
    except Exception as e:
        print(f"  Ollama (direct): FAILED - {e}")
        sys.exit(1)

    try:
        httpx.get(f"{PROXY_URL}/api/tags", timeout=5).raise_for_status()
        print(f"  Memory proxy:    OK")
    except Exception as e:
        print(f"  Memory proxy:    FAILED - {e}")
        sys.exit(1)

    # --- Phase 1: Seed all facts through the PROXY (builds memory) ---
    print(f"\n{'='*70}")
    print("PHASE 1: Seeding facts through memory proxy")
    print(f"{'='*70}")
    all_seeds = []
    for test in TESTS:
        for seed in test["seeds"]:
            if seed not in all_seeds:
                all_seeds.append(seed)

    print(f"  Seeding {len(all_seeds)} unique fact messages...")
    seed_latencies = []
    for i, seed in enumerate(all_seeds, 1):
        lat = seed_fact(PROXY_URL, seed)
        seed_latencies.append(lat)
        print(f"    [{i}/{len(all_seeds)}] {seed[:60]}... ({lat:.0f}ms)")

    print(f"  Waiting {SETTLING_DELAY}s for background storage...")
    time.sleep(SETTLING_DELAY)
    print(f"  Seeding complete. Mean latency: {statistics.mean(seed_latencies):.0f}ms")

    # --- Phase 2: Run queries against DIRECT Ollama (no memory) ---
    print(f"\n{'='*70}")
    print("PHASE 2: Querying DIRECT Ollama (no memory)")
    print(f"{'='*70}")
    direct_run = BenchmarkRun(mode="direct", url=DIRECT_URL, model=MODEL)
    t0 = time.perf_counter()
    total = len(TESTS)
    for i, test in enumerate(TESTS, 1):
        tid = test["id"]
        print(f"  [{i}/{total}] {tid} ...", end=" ", flush=True)
        response, query_lat = query_fresh(DIRECT_URL, test["query"])
        found, missing, sc = score_response(response, test["expected"])
        pf = pass_fail(sc)
        direct_run.results.append(TestResult(
            test_id=tid, category=test["category"], query=test["query"],
            expected_keywords=test["expected"], response=response[:500],
            keywords_found=found, keywords_missing=missing,
            score=sc, latency_ms=query_lat, pass_fail=pf,
        ))
        print(f"{pf} (score={sc:.0%}, {query_lat:.0f}ms)")
    direct_run.total_time_s = time.perf_counter() - t0

    # --- Phase 3: Run queries against PROXY (with memory) ---
    print(f"\n{'='*70}")
    print("PHASE 3: Querying MEMORY PROXY (with memory)")
    print(f"{'='*70}")
    proxy_run = BenchmarkRun(mode="proxy", url=PROXY_URL, model=MODEL)
    t0 = time.perf_counter()
    for i, test in enumerate(TESTS, 1):
        tid = test["id"]
        print(f"  [{i}/{total}] {tid} ...", end=" ", flush=True)
        response, query_lat = query_fresh(PROXY_URL, test["query"])
        found, missing, sc = score_response(response, test["expected"])
        pf = pass_fail(sc)
        proxy_run.results.append(TestResult(
            test_id=tid, category=test["category"], query=test["query"],
            expected_keywords=test["expected"], response=response[:500],
            keywords_found=found, keywords_missing=missing,
            score=sc, latency_ms=query_lat, pass_fail=pf,
        ))
        print(f"{pf} (score={sc:.0%}, {query_lat:.0f}ms)")
    proxy_run.total_time_s = time.perf_counter() - t0

    # --- Phase 4: Latency benchmark ---
    print(f"\n{'='*70}")
    print("PHASE 4: Latency Overhead Measurement")
    print(f"{'='*70}")
    lat_direct = run_latency_benchmark(DIRECT_URL, "direct", n_rounds=5)
    lat_proxy = run_latency_benchmark(PROXY_URL, "proxy", n_rounds=5)

    # --- Phase 5: Generate report ---
    print(f"\n{'='*70}")
    print("PHASE 5: Generating Report")
    print(f"{'='*70}")

    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "BENCHMARK_REPORT.md"
    )
    report = generate_report(direct_run, proxy_run, lat_direct, lat_proxy, report_path)

    # Print summary to console
    d_avg = statistics.mean([r.score for r in direct_run.results]) * 100
    p_avg = statistics.mean([r.score for r in proxy_run.results]) * 100
    d_pass = sum(1 for r in direct_run.results if r.pass_fail == "PASS")
    p_pass = sum(1 for r in proxy_run.results if r.pass_fail == "PASS")

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Direct (no memory):  {d_avg:.1f}% avg score, {d_pass}/{len(TESTS)} passed")
    print(f"  Memory Proxy:        {p_avg:.1f}% avg score, {p_pass}/{len(TESTS)} passed")
    print(f"  Improvement:         +{p_avg - d_avg:.1f}% score")
    print(f"  Latency overhead:    +{lat_proxy['mean_ms'] - lat_direct['mean_ms']:.0f}ms mean")
    print(f"  Report saved to:     {report_path}")
    print(f"{'='*70}")

    # Also save raw JSON
    json_path = report_path.replace(".md", ".json")
    raw = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "model": MODEL,
        "direct": {
            "avg_score": d_avg,
            "passed": d_pass,
            "total": len(TESTS),
            "results": [
                {"id": r.test_id, "score": r.score, "pass_fail": r.pass_fail,
                 "latency_ms": r.latency_ms, "response": r.response}
                for r in direct_run.results
            ],
        },
        "proxy": {
            "avg_score": p_avg,
            "passed": p_pass,
            "total": len(TESTS),
            "results": [
                {"id": r.test_id, "score": r.score, "pass_fail": r.pass_fail,
                 "latency_ms": r.latency_ms, "response": r.response,
                 "keywords_found": r.keywords_found, "keywords_missing": r.keywords_missing}
                for r in proxy_run.results
            ],
        },
        "latency": {"direct": lat_direct, "proxy": lat_proxy},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print(f"  Raw JSON saved to:   {json_path}")


if __name__ == "__main__":
    main()
