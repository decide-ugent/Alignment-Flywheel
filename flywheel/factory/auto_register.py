"""auto_register — imports every implementation file and registers each class.

Adding a new implementation: write the file, then add one line here.
"""

from flywheel.factory.registry import register_class


def register_all() -> None:
    """Import every concrete implementation and register it by class name."""

    # ── Red Team — observe ─────────────────────────────────────
    from flywheel.roles.redteam.observe.grid_observer import GridObserver
    from flywheel.roles.redteam.observe.fine_sampling_observer import FineSamplingObserver
    from flywheel.roles.redteam.observe.medical_case_generator import MedicalCaseGenerator
    register_class("GridObserver", GridObserver)
    register_class("FineSamplingObserver", FineSamplingObserver)
    register_class("MedicalCaseGenerator", MedicalCaseGenerator)

    # ── Red Team — orient ──────────────────────────────────────
    from flywheel.roles.redteam.orient.distance_orienter import DistanceOrienter
    from flywheel.roles.redteam.orient.medical_case_orienter import MedicalCaseOrienter
    register_class("DistanceOrienter", DistanceOrienter)
    register_class("MedicalCaseOrienter", MedicalCaseOrienter)

    # ── Red Team — decide ──────────────────────────────────────
    from flywheel.roles.redteam.decide.farthest_first_decider import FarthestFirstDecider
    from flywheel.roles.redteam.decide.random_decider import RandomDecider
    from flywheel.roles.redteam.decide.medical_priority_decider import MedicalPriorityDecider
    register_class("FarthestFirstDecider", FarthestFirstDecider)
    register_class("RandomDecider", RandomDecider)
    register_class("MedicalPriorityDecider", MedicalPriorityDecider)

    # ── Red Team — act ─────────────────────────────────────────
    from flywheel.roles.redteam.act.candidate_submitter import CandidateSubmitter
    from flywheel.roles.redteam.act.medical_candidate_submitter import MedicalCandidateSubmitter
    register_class("CandidateSubmitter", CandidateSubmitter)
    register_class("MedicalCandidateSubmitter", MedicalCandidateSubmitter)

    # ── Verifier — observe ─────────────────────────────────────
    from flywheel.roles.verifier.observe.norm_loader import NormLoader
    register_class("NormLoader", NormLoader)

    # ── Verifier — orient ──────────────────────────────────────
    from flywheel.roles.verifier.orient.spatial_norm_matcher import SpatialNormMatcher
    from flywheel.roles.verifier.orient.medical_norm_matcher import MedicalNormMatcher
    register_class("SpatialNormMatcher", SpatialNormMatcher)
    register_class("MedicalNormMatcher", MedicalNormMatcher)

    # ── Verifier — decide ──────────────────────────────────────
    from flywheel.roles.verifier.decide.spatial_violation_decider import SpatialViolationDecider
    from flywheel.roles.verifier.decide.medical_violation_decider import MedicalViolationDecider
    register_class("SpatialViolationDecider", SpatialViolationDecider)
    register_class("MedicalViolationDecider", MedicalViolationDecider)

    # ── Verifier — act ─────────────────────────────────────────
    from flywheel.roles.verifier.act.verification_emitter import VerificationEmitter
    register_class("VerificationEmitter", VerificationEmitter)

    # ── Refinement — observe ───────────────────────────────────
    from flywheel.roles.refinement.observe.queue_observer import QueueObserver
    from flywheel.roles.refinement.observe.medical_queue_observer import MedicalQueueObserver
    register_class("QueueObserver", QueueObserver)
    register_class("MedicalQueueObserver", MedicalQueueObserver)

    # ── Refinement — orient ────────────────────────────────────
    from flywheel.roles.refinement.orient.adaptive_bandwidth_orienter import AdaptiveBandwidthOrienter
    from flywheel.roles.refinement.orient.fixed_bandwidth_orienter import FixedBandwidthOrienter
    from flywheel.roles.refinement.orient.medical_correction_orienter import MedicalCorrectionOrienter
    register_class("AdaptiveBandwidthOrienter", AdaptiveBandwidthOrienter)
    register_class("FixedBandwidthOrienter", FixedBandwidthOrienter)
    register_class("MedicalCorrectionOrienter", MedicalCorrectionOrienter)

    # ── Refinement — decide ────────────────────────────────────
    from flywheel.roles.refinement.decide.cumulative_regression_decider import CumulativeRegressionDecider
    from flywheel.roles.refinement.decide.no_cumulative_decider import NoCumulativeDecider
    from flywheel.roles.refinement.decide.predictive_coverage_decider import PredictiveCoverageDecider
    from flywheel.roles.refinement.decide.medical_batch_decider import MedicalBatchDecider
    register_class("CumulativeRegressionDecider", CumulativeRegressionDecider)
    register_class("NoCumulativeDecider", NoCumulativeDecider)
    register_class("PredictiveCoverageDecider", PredictiveCoverageDecider)
    register_class("MedicalBatchDecider", MedicalBatchDecider)

    # ── Refinement — act ───────────────────────────────────────
    from flywheel.roles.refinement.act.batch_deployer import BatchDeployer
    from flywheel.roles.refinement.act.medical_batch_deployer import MedicalBatchDeployer
    register_class("BatchDeployer", BatchDeployer)
    register_class("MedicalBatchDeployer", MedicalBatchDeployer)

    # ── Triage ─────────────────────────────────────────────────
    from flywheel.roles.triage.fifo_triage import FIFOTriage
    from flywheel.roles.triage.priority_triage import PriorityTriage
    register_class("FIFOTriage", FIFOTriage)
    register_class("PriorityTriage", PriorityTriage)

    # ── Blue Team ──────────────────────────────────────────────
    from flywheel.roles.blueteam.collateral_monitor import CollateralMonitor
    register_class("CollateralMonitor", CollateralMonitor)

    # ── Oracles ────────────────────────────────────────────────
    from flywheel.roles.oracle.spatial_oracle import SpatialOracle
    from flywheel.roles.oracle.patient_portal_oracle import PatientPortalOracle
    from flywheel.roles.oracle.simple_medical_oracle import SimpleMedicalOracle
    from flywheel.roles.oracle.complex_medical_oracle import ComplexMedicalOracle
    from flywheel.roles.oracle.adapters.precomputed_grid_oracle import PrecomputedGridOracle
    register_class("SpatialOracle", SpatialOracle)
    register_class("PatientPortalOracle", PatientPortalOracle)
    register_class("SimpleMedicalOracle", SimpleMedicalOracle)
    register_class("ComplexMedicalOracle", ComplexMedicalOracle)
    register_class("PrecomputedGridOracle", PrecomputedGridOracle)

    # ── Flywheel overlays ──────────────────────────────────────
    from flywheel.roles.flywheel_overlay.spatial_overlay import SpatialOverlay
    from flywheel.roles.flywheel_overlay.patient_portal_overlay import PatientPortalOverlay
    from flywheel.roles.flywheel_overlay.simple_medical_overlay import SimpleMedicalOverlay
    from flywheel.roles.flywheel_overlay.complex_medical_overlay import ComplexMedicalOverlay
    register_class("SpatialOverlay", SpatialOverlay)
    register_class("PatientPortalOverlay", PatientPortalOverlay)
    register_class("SimpleMedicalOverlay", SimpleMedicalOverlay)
    register_class("ComplexMedicalOverlay", ComplexMedicalOverlay)

    # ── Enforcement ────────────────────────────────────────────
    from flywheel.roles.enforcement.default_enforcement import DefaultEnforcement
    register_class("DefaultEnforcement", DefaultEnforcement)

    # ── Proposer ───────────────────────────────────────────────
    from flywheel.roles.proposer.passthrough_proposer import PassthroughProposer
    from flywheel.roles.proposer.spatial_proposer import SpatialProposer
    register_class("PassthroughProposer", PassthroughProposer)
    register_class("SpatialProposer", SpatialProposer)

    # ── Knowledge Base ─────────────────────────────────────────
    from flywheel.core.knowledge_base.in_memory_knowledge_base import InMemoryKnowledgeBase
    register_class("InMemoryKnowledgeBase", InMemoryKnowledgeBase)
