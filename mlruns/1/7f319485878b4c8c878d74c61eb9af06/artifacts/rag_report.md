# RAG Execution Report

**Run ID:** `7f319485878b4c8c878d74c61eb9af06`
**Query:** What is the platform monthly uptime guarantee and what are the service credit tiers?
**Model:** extractive-offline-v1
**Retrieval Mode:** hybrid

## Generated Response
**Executive Summary:**
Based on the enterprise repository analysis for query *"What is the platform monthly uptime guarantee and what are the service credit tiers?"*, the following key points were retrieved:

**Key Findings & Policy Terms:**
- SERVICE CREDIT CALCULATION
If platform uptime falls below the commitment during any calendar month, the Customer shall be entitled to the following SLA credits:
- 99.0% to 99.94% Uptime: 10% credit of monthly billing fees. [Source: customer_contracts_sla.txt | Chunk: doc_b125793155ca67e9_c001]
- SYSTEM UPTIME GUARANTEE
The Service Provider guarantees monthly platform uptime of 99.95% ("Uptime Commitment"), excluding scheduled maintenance windows announced at least 72 hours in advance. [Source: customer_contracts_sla.txt | Chunk: doc_b125793155ca67e9_c000]
- - Below 95.0% Uptime: 50% credit of monthly billing fees and right to immediate contract termination without penalty. [Source: customer_contracts_sla.txt | Chunk: doc_b125793155ca67e9_c001]
- Executive Summary
This document specifies the target state cloud infrastructure for our global SaaS platform supporting AI-driven document intelligence, real-time analytics, and transactional workloads across AWS and Google Cloud Platform. [Source: cloud_architecture_spec.md | Chunk: doc_bc9f2909b38a48c0_c000]
- - 95.0% to 98.99% Uptime: 25% credit of monthly billing fees. [Source: customer_contracts_sla.txt | Chunk: doc_b125793155ca67e9_c001]
- MASTER SERVICE AGREEMENT - SERVICE LEVEL ADDENDUM (SLA)
Document ID: LEGAL-SLA-2025
Applicable Tier: Platinum & Enterprise Tier Customers 1. [Source: customer_contracts_sla.txt | Chunk: doc_b125793155ca67e9_c000]

*Response synthesized with source verification and strict hallucination guardrails.*

## Retrieved Context Chunks (3)
### Snippet 1: customer_contracts_sla.txt (ID: doc_b125793155ca67e9_c001)
- Score: 0.0164
- Content:
> 2. SERVICE CREDIT CALCULATION
If platform uptime falls below the commitment during any calendar month, the Customer shall be entitled to the following SLA credits:
- 99.0% to 99.94% Uptime: 10% credit of monthly billing fees.
- 95.0% to 98.99% Uptime: 25% credit of monthly billing fees.
- Below 95.0% Uptime: 50% credit of monthly billing fees and right to immediate contract termination without penalty. 3. INCIDENT RESPONSE TIERS & SEVERITY LEVELS

### Snippet 2: customer_contracts_sla.txt (ID: doc_b125793155ca67e9_c000)
- Score: 0.0161
- Content:
> MASTER SERVICE AGREEMENT - SERVICE LEVEL ADDENDUM (SLA)
Document ID: LEGAL-SLA-2025
Applicable Tier: Platinum & Enterprise Tier Customers 1. SYSTEM UPTIME GUARANTEE
The Service Provider guarantees monthly platform uptime of 99.95% ("Uptime Commitment"), excluding scheduled maintenance windows announced at least 72 hours in advance.

### Snippet 3: cloud_architecture_spec.md (ID: doc_bc9f2909b38a48c0_c000)
- Score: 0.0155
- Content:
> # Multi-Region Enterprise Cloud Architecture Specification
Document ID: ARCH-SPEC-884
Architectural Status: Approved
Target Availability: 99.99% ## 1. Executive Summary
This document specifies the target state cloud infrastructure for our global SaaS platform supporting AI-driven document intelligence, real-time analytics, and transactional workloads across AWS and Google Cloud Platform.

