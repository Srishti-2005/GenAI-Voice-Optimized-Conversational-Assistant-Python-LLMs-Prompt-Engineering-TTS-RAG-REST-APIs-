# RAG Execution Report

**Run ID:** `7fa228808bfe4233bd7dffe7920291ee`
**Query:** What are the latency SLAs for P95 API gateway and the disaster recovery RTO / RPO?
**Model:** extractive-offline-v1
**Retrieval Mode:** hybrid

## Generated Response
**Executive Summary:**
Based on the enterprise repository analysis for query *"What are the latency SLAs for P95 API gateway and the disaster recovery RTO / RPO?"*, the following key points were retrieved:

**Key Findings & Policy Terms:**
- Latency SLAs and Network Topology
- API gateway response latency: P95 < 120ms; P99 < 350ms. [Source: cloud_architecture_spec.md | Chunk: doc_bc9f2909b38a48c0_c003]
- APPROVED LLM & API USAGE - All external API calls to public LLMs (e.g., OpenAI, Anthropic, Google Cloud Vertex AI) must route through the enterprise API Gateway proxy. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c001]
- Disaster Recovery (DR) & Failover
- Recovery Time Objective (RTO): < 15 minutes. [Source: cloud_architecture_spec.md | Chunk: doc_bc9f2909b38a48c0_c004]
- - End-to-end RAG query pipeline (retrieval + reranking + LLM first token latency): P95 < 1.8 seconds. [Source: cloud_architecture_spec.md | Chunk: doc_bc9f2909b38a48c0_c003]
- - Cloudflare Enterprise CDN handles edge caching, DDoS mitigation, and SSL termination. [Source: cloud_architecture_spec.md | Chunk: doc_bc9f2909b38a48c0_c003]
- - Zero Data Retention (ZDR) agreements must be active for all proprietary and customer-facing interactions. [Source: enterprise_ai_policy.txt | Chunk: doc_25223dec50e8bc17_c001]

*Response synthesized with source verification and strict hallucination guardrails.*

## Retrieved Context Chunks (3)
### Snippet 1: cloud_architecture_spec.md (ID: doc_bc9f2909b38a48c0_c003)
- Score: 0.0164
- Content:
> - **Blob Storage**: Object storage (S3 / GCS) configured with lifecycle policies archiving raw document uploads to cold storage after 90 days. ## 4. Latency SLAs and Network Topology
- API gateway response latency: P95 < 120ms; P99 < 350ms.
- End-to-end RAG query pipeline (retrieval + reranking + LLM first token latency): P95 < 1.8 seconds.
- Cross-region database replication latency: < 500ms.
- Cloudflare Enterprise CDN handles edge caching, DDoS mitigation, and SSL termination.

### Snippet 2: enterprise_ai_policy.txt (ID: doc_25223dec50e8bc17_c001)
- Score: 0.0160
- Content:
> 2. APPROVED LLM & API USAGE - All external API calls to public LLMs (e.g., OpenAI, Anthropic, Google Cloud Vertex AI) must route through the enterprise API Gateway proxy. - Direct connection from client devices or non-isolated VPCs to external model endpoints is strictly prohibited. - Zero Data Retention (ZDR) agreements must be active for all proprietary and customer-facing interactions.

### Snippet 3: cloud_architecture_spec.md (ID: doc_bc9f2909b38a48c0_c004)
- Score: 0.0160
- Content:
> ## 5. Disaster Recovery (DR) & Failover
- Recovery Time Objective (RTO): < 15 minutes.
- Recovery Point Objective (RPO): < 1 minute.
- Automated health probes evaluate regional DNS health every 10 seconds; Route 53 failover initiates automatically if 3 consecutive health checks fail.

