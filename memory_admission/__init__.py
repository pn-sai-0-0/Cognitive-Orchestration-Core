"""Memory Admission Policy for COC v3.

The single responsibility of this package is to decide **whether** a
candidate memory should be written to long-term memory. It does NOT:

  * change retrieval, ranking, embeddings, or recall in any way;
  * modify existing memory rows or their scoring;
  * introduce new memory kinds, indices, or storage back-ends;
  * implement Learning, Adaptation, Planning, or Replay.

It gates writes only. Retrieval must continue to behave exactly as
before.
"""

from memory_admission.contracts import AdmissionDecision, MemoryCandidate
from memory_admission.policy import AdmissionPolicy

__all__ = [
    "AdmissionDecision",
    "AdmissionPolicy",
    "MemoryCandidate",
]
