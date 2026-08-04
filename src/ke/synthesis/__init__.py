"""Intentional-stance cross-ontology synthesis."""

from .align import align_ontologies, align_ontologies_with_gateway
from .critic import critique_frame, repair_frame, review_and_repair
from .frame import compose_bdi_frame, compose_bdi_frame_with_gateway
from .integrate import build_integrated_named_graph, identify_conflicts
from .models import AlignmentReport, IntegratedOntology, ReviewReport, SynthesisBundle
from .render import compile_system_prompt

__all__ = [
    "AlignmentReport",
    "IntegratedOntology",
    "ReviewReport",
    "SynthesisBundle",
    "align_ontologies",
    "align_ontologies_with_gateway",
    "build_integrated_named_graph",
    "compile_system_prompt",
    "compose_bdi_frame",
    "compose_bdi_frame_with_gateway",
    "critique_frame",
    "identify_conflicts",
    "repair_frame",
    "review_and_repair",
]
