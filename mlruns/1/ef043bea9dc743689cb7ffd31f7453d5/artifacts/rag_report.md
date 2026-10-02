# RAG Execution Report

**Run ID:** `ef043bea9dc743689cb7ffd31f7453d5`
**Query:** What are the security guidelines regarding external LLM API proxies and customer PII?
**Model:** extractive-offline-v1
**Retrieval Mode:** hybrid

## Generated Response
**Executive Summary:**
Based on the enterprise repository analysis for query *"What are the security guidelines regarding external LLM API proxies and customer PII?"*, the following key points were retrieved:

**Key Findings & Policy Terms:**
- APPROVED LLM & API USAGE - All external API calls to public LLMs (e.g., OpenAI, Anthropic, Google Cloud Vertex AI) must route through the enterprise API Gateway proxy. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c001]
- PURPOSE & SCOPE
This document outlines mandatory guidelines for utilizing Generative Artificial Intelligence (GenAI), Large Language Models (LLMs), and automated data retrieval systems across all company business units and engineering teams. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c000]
- - Zero Data Retention (ZDR) agreements must be active for all proprietary and customer-facing interactions. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c001]
- APPROVED LLM & API USAGE [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c000]
- | Q2-2025   | Engineering    | Fine-Tuning & Prompt Lab   |                 300000 |             285000 |              195 | Completed | | Q2-2025   | Security       | PII Redaction & Compliance |                 150000 |             142000 |              160 | Completed | | Q3-2025   | Engineering    | Enterprise Multi-Agent RAG |                 450000 |             420000 |              240 | Active    | [Source: quarterly_financial_report.csv | Chunk: doc_7099f8a1827c905f_c002]
- - Direct connection from client devices or non-isolated VPCs to external model endpoints is strictly prohibited. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c001]

*Response synthesized with source verification and strict hallucination guardrails.*

## Retrieved Context Chunks (3)
### Snippet 1: enterprise_ai_policy.txt (ID: doc_25223dec50e8bc17_c001)
- Score: 0.0164
- Content:
> 2. APPROVED LLM & API USAGE - All external API calls to public LLMs (e.g., OpenAI, Anthropic, Google Cloud Vertex AI) must route through the enterprise API Gateway proxy. - Direct connection from client devices or non-isolated VPCs to external model endpoints is strictly prohibited. - Zero Data Retention (ZDR) agreements must be active for all proprietary and customer-facing interactions.

### Snippet 2: enterprise_ai_policy.txt (ID: doc_25223dec50e8bc17_c000)
- Score: 0.0161
- Content:
> # Global Enterprise AI Governance & Security Policy (v3.2)
Document ID: SEC-POL-2025-09
Classification: Internal Confidential
Effective Date: January 1, 2025
Review Cycle: Annual 1. PURPOSE & SCOPE
This document outlines mandatory guidelines for utilizing Generative Artificial Intelligence (GenAI), Large Language Models (LLMs), and automated data retrieval systems across all company business units and engineering teams. 2. APPROVED LLM & API USAGE

### Snippet 3: quarterly_financial_report.csv (ID: doc_7099f8a1827c905f_c002)
- Score: 0.0155
- Content:
> | Q2-2025   | Engineering    | Fine-Tuning & Prompt Lab   |                 300000 |             285000 |              195 | Completed | | Q2-2025   | Security       | PII Redaction & Compliance |                 150000 |             142000 |              160 | Completed | | Q3-2025   | Engineering    | Enterprise Multi-Agent RAG |                 450000 |             420000 |              240 | Active    |

