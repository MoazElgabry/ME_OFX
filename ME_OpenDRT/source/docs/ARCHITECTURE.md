# ARCHITECTURE.md — OFX Plugin Architecture

This repository follows a layered design to keep OFX host logic, algorithms, and compute backends clearly separated.

## Layer 1 — Host Interface

Handles:

- plugin descriptors
- parameters
- clips
- instance lifecycle
- render entry

No heavy processing should occur here.

## Layer 2 — Render Orchestration

Responsibilities:

- extract parameters
- build render state
- validate clips
- dispatch to backend

## Layer 3 — Core Algorithm

Pure processing logic.

Properties:

- host independent
- deterministic
- testable

## Layer 4 — Backend Implementations

Possible backends:

- CPU
- Metal
- CUDA
- OpenCL

Each backend implements the same algorithm behaviour.

## Layer 5 — Diagnostics

Includes:

- debug logging
- backend selection reporting
- performance timing
- validation checks

## Data Flow

Host image → Render state → Core algorithm → Backend execution → Output image
