# RAG Execution Report

**Run ID:** `552d8434c64541e9a504e7591609b570`
**Query:** What is the policy for external LLM API calls and PII?
**Model:** extractive-offline-v1
**Retrieval Mode:** hybrid

## Generated Response
**Executive Summary:**
Based on the enterprise repository analysis for query *"What is the policy for external LLM API calls and PII?"*, the following key points were retrieved:

**Key Findings & Policy Terms:**
- APPROVED LLM & API USAGE - All external API calls to public LLMs (e.g., OpenAI, Anthropic, Google Cloud Vertex AI) must route through the enterprise API Gateway proxy. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c002]
- CONTACTS & ESCALATIONS
For policy exceptions or new AI model onboarding, submit a ticket to security-governance@enterprise.internal or contact the Chief Information Security Officer (CISO). [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c009]
- - Direct connection from client devices or non-isolated VPCs to external model endpoints is strictly prohibited. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c002]

*Response synthesized with source verification and strict hallucination guardrails.*

## Retrieved Context Chunks (2)
### Snippet 1: enterprise_ai_policy.txt (ID: doc_25223dec50e8bc17_c002)
- Score: 0.0164
- Content:
> 2. APPROVED LLM & API USAGE - All external API calls to public LLMs (e.g., OpenAI, Anthropic, Google Cloud Vertex AI) must route through the enterprise API Gateway proxy. - Direct connection from client devices or non-isolated VPCs to external model endpoints is strictly prohibited.

### Snippet 2: enterprise_ai_policy.txt (ID: doc_25223dec50e8bc17_c009)
- Score: 0.0160
- Content:
> 6. CONTACTS & ESCALATIONS
For policy exceptions or new AI model onboarding, submit a ticket to security-governance@enterprise.internal or contact the Chief Information Security Officer (CISO).

