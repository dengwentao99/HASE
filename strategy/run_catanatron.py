from catanatron_common import MethodProfile, main_for_profile


PROFILE = MethodProfile(
    method_name="HASE-v1",
    player_code="HASEV1",
    paradigm="Hierarchical adversarial self-evolution: dynamic macro role selection plus same-model mirror micro-optimization.",
    macro_enabled=True,
    micro_enabled=True,
    curriculum_mode="easy_to_hard",
    early_stop=True,
    stop_reliability=0.92,
    curriculum_constraints=True,
    prompt_focus=(
        "Use strict easy-to-hard curriculum evolution: easy rounds stabilize opening expansion, "
        "medium rounds add midgame conversion, and hard rounds add adversarial/endgame optimization. "
        "Macro selects the round direction; micro mirror feedback performs constrained step-level refinements."
    ),
)


if __name__ == "__main__":
    main_for_profile(PROFILE, "results-catanatron/HASE-v1-catanatron-seed100")
