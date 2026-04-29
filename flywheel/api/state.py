"""api/state — module-level holders for component instances.

The Flask blueprints reach into here to find the configured
oracle/proposer/flywheel/enforcement instances.  Set by app
startup.
"""

oracle = None         # BaseOracle or BaseSpatialOracleAdapter wrapper
proposer = None       # BaseProposer
flywheel = None       # BaseFlywheelOverlay
enforcement = None    # BaseEnforcementPolicy


def set_components(oracle_, proposer_, flywheel_, enforcement_):
    global oracle, proposer, flywheel, enforcement
    oracle = oracle_
    proposer = proposer_
    flywheel = flywheel_
    enforcement = enforcement_
