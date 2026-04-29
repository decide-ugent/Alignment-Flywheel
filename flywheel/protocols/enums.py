"""Canonical enumerations shared across the framework."""

from enum import Enum


class TrajectoryKind(str, Enum):
    ACTION = "action"
    TOOL_CALL = "tool_call"
    MESSAGE = "message"
    PLAN = "plan"
    SPATIAL = "spatial"


class EnforcementAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


class AuditStatus(str, Enum):
    COVERED = "covered"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"


class CorrectionType(str, Enum):
    SPATIAL_FLAW_PATCH = "spatial_flaw_patch"
    AUDIT_COVERAGE_UPDATE = "audit_coverage_update"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    MEDICAL_HARD_BLOCK = "medical_hard_block"
    NORM_UPDATE = "norm_update"


class VerificationOutcome(str, Enum):
    VIOLATION = "violation"
    NO_VIOLATION = "no_violation"
    INCONCLUSIVE = "inconclusive"


class NormKind(str, Enum):
    KEYWORD_BLOCK = "keyword_block"
    REGEX = "regex"
    SPATIAL_BOUNDARY = "spatial_boundary"
    PREDICATE = "predicate"
    THRESHOLD_RULE = "threshold_rule"
