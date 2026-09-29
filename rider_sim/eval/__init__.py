"""Fidelity evaluation of simulated rider agents against real behavior."""

from rider_sim.eval.ablations import CONFIGS, permutation_matrix
from rider_sim.eval.discriminator import run_discriminator
from rider_sim.eval.runner import run_evaluation

__all__ = ["CONFIGS", "permutation_matrix", "run_discriminator", "run_evaluation"]
