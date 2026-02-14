# Ollama Memory Proxy - Benchmark Report

**Date:** 2026-02-14 16:15:53
**Model:** qwen3:0.6b
**Direct Ollama:** http://127.0.0.1:11434
**Memory Proxy:** http://127.0.0.1:11435
**Total tests:** 14

## Executive Summary

| Metric | Direct (no memory) | Memory Proxy | Delta |
|--------|-------------------|-------------|-------|
| **Average Score** | 11.9% | 78.6% | **+66.7%** |
| Tests PASSED | 0/14 | 10/14 | +10 |
| Tests FAILED | 12/14 | 2/14 | -10 |
| Mean Query Latency | 775ms | 735ms | +-39ms |
| Total Runtime | 18.3s | 21.3s | |

## Per-Category Breakdown

| Category | Direct Avg | Proxy Avg | Improvement |
|----------|-----------|----------|-------------|
| 1. Personal Fact Recall | 10% | 100% | **+90%** |
| 2. Multi-Turn Continuity | 11% | 50% | **+39%** |
| 3. Temporal Context | 25% | 75% | **+50%** |
| 4. Cross-Topic Association | 17% | 50% | **+33%** |
| 5. Precision Recall | 0% | 100% | **+100%** |

## Detailed Test Results

| Test ID | Category | Direct | Proxy | Keywords Found (Proxy) |
|---------|----------|--------|-------|----------------------|
| PF-01-name-age | 1. | FAIL (0%) | PASS (100%) | Lucian, 35 |
| PF-02-location-job | 1. | FAIL (0%) | PASS (100%) | Timisoara, Romania, AI |
| PF-03-preferences | 1. | PART (50%) | PASS (100%) | Python, Neovim |
| PF-04-pets | 1. | FAIL (0%) | PASS (100%) | Shadow, Rex, cat, retriever |
| PF-05-contact | 1. | FAIL (0%) | PASS (100%) | 0722, 555, 123, lucian@example.com |
| MT-01-project-hw | 2. | FAIL (33%) | PASS (100%) | drone, Raspberry Pi, camera |
| MT-02-medical | 2. | FAIL (0%) | FAIL (0%) | - |
| MT-03-books | 2. | FAIL (0%) | PART (50%) | Godel, Hofstadter |
| TC-01-laptop-update | 3. | PART (50%) | PASS (100%) | ThinkPad, 64 |
| TC-02-rent-update | 3. | FAIL (0%) | PART (50%) | Victoriei |
| CT-01-sister | 4. | FAIL (0%) | PASS (100%) | Elena, doctor, Cluj, March, 15 |
| CT-02-car | 4. | FAIL (33%) | FAIL (0%) | - |
| PR-01-wifi | 5. | FAIL (0%) | PASS (100%) | BorbeNet5G, Tr0ub4dor |
| PR-02-server | 5. | FAIL (0%) | PASS (100%) | 192.168.1.42, 8080, lucian_admin, PostgreSQL |

## Latency Analysis

| Metric | Direct | Proxy | Overhead |
|--------|--------|-------|----------|
| Mean | 775ms | 735ms | +-39ms |
| Median | 600ms | 664ms | +63ms |
| Min | 589ms | 582ms | |
| Max | 1330ms | 928ms | |
| **Overhead %** | - | - | **-5.0%** |

## Sample Responses (Selected Tests)

### PF-01-name-age: What is my name and how old am I?

**Expected keywords:** Lucian, 35

**Direct (score: 0%):**
> I can't answer without knowing your name and age. Could you please provide those details?

**Proxy (score: 100%):**
> Your name is **Lucian Borbeleac**, and you are **35 years old**. Let me know if you need further assistance! 😊

### MT-01-project-hw: What project am I working on and what hardware does it use?

**Expected keywords:** drone, Raspberry Pi, camera

**Direct (score: 33%):**
> To answer your question, I need to know:  
1. **What project are you working on?**  
2. **What hardware does it use?**  

Could you please provide these details? For example:  
- "I’m working on a project to develop a mobile app."  
- "I’m using a Raspberry Pi for my project."  

Once you share this

**Proxy (score: 100%):**
> Your project is a drone delivery system that uses a Raspberry Pi 5 as its flight controller. The system includes a stereo camera for computer vision, enabling autonomous navigation and package delivery. Let me know if you'd like further details!

### CT-01-sister: What does my sister do for work, where does she live, and when is she getting married?

**Expected keywords:** Elena, doctor, Cluj, March, 15

**Direct (score: 0%):**
> I can't provide specific information about your sister's work, where she lives, or when she's getting married. However, I'd be happy to help with other questions or share general information about life! Let me know how I can assist you. 😊

**Proxy (score: 100%):**
> Your sister, Elena, works as a doctor at the **county hospital** and lives in **Cluj-Napoca**. She is getting married on **March 15th**. Let me know if you need further assistance! 🎉

### PR-02-server: What is my server IP, port, admin username, and database engine?

**Expected keywords:** 192.168.1.42, 8080, lucian_admin, PostgreSQL

**Direct (score: 0%):**
> To provide accurate information, please clarify the following details:

1. **Server IP**: If you're running a server, please specify the IP address.
2. **Server Port**: If you're running a server, please specify the port number.
3. **Admin Username**: If you're using an admin account, please provide

**Proxy (score: 100%):**
> Your server's IP address is **192.168.1.42**, it runs on port **8080**, the admin username is **lucian_admin**, and the database engine is **PostgreSQL 16**. Let me know if you need further assistance! 😊

## Methodology

1. **Seed phase:** Facts are sent as individual chat messages to establish memory context
2. **Query phase:** Questions are asked in a **brand-new conversation** (no chat history)
3. **Scoring:** Keyword presence check (case-insensitive) against expected facts
4. **Pass criteria:** >=80% keywords found = PASS, >=40% = PARTIAL, <40% = FAIL
5. **Direct mode:** Queries go straight to Ollama (no memory) - LLM has no prior context
6. **Proxy mode:** Queries go through memory proxy which injects relevant past conversations
7. **Latency:** Measured separately with minimal queries to isolate proxy overhead
8. **Temperature:** 0.1 for reproducibility
