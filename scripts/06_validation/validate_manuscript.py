#!/usr/bin/env python3
"""Fail fast if the corrected manuscript drifts from its verified artifacts."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "results"


def assert_in_order(text, markers):
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)


def main():
    dataset = pd.read_csv(RESULTS / "dataset_audit.csv").iloc[0]
    assert dataset["n_total"] == 94
    assert dataset["n_YS"] == 93
    assert dataset["n_HV"] == 94
    assert dataset["n_unique_compositions_all"] == 82
    assert dataset["n_unique_compositions_YS"] == 81
    assert dataset["C_campaign_hold_min_h"] == 1.0
    assert dataset["n_compositions_crossing_batches"] == 1

    pca = pd.read_csv(RESULTS / "pca_ols_nested_results.csv").set_index("protocol")
    assert abs(pca.loc["LOO", "R2"] - 0.4800498425) < 1e-8
    assert abs(pca.loc["LOBO", "R2"] - 0.3619266847) < 1e-8

    hardness = pd.read_csv(RESULTS / "hardness_equation_corrected.csv")
    full = hardness[hardness["protocol"] == "full fit"].iloc[0]
    assert full["c1"] > 0
    assert full["c2"] < 0
    assert abs(full["R2"] - 0.7439184430) < 1e-8
    lobo = hardness[hardness["protocol"] == "LOBO"].iloc[0]
    assert abs(lobo["R2"] - 0.6356118501) < 1e-8

    external = pd.read_csv(RESULTS / "external_validation_by_evidence_tier.csv")
    measured = external[external["target_status"] == "direct YS measurement"]
    assert len(measured) == 3
    assert (measured["R2"] < 0).all()

    grouped = pd.read_csv(RESULTS / "main2_grouped_validation.csv")
    assert set(grouped["protocol"]) == {"5-fold", "LOO", "LOBO"}
    m15 = grouped[(grouped["model"] == "F3 M15 SD-grain interaction")].set_index("protocol")
    assert abs(m15.loc["LOO", "R2"] - 0.6943983232) < 1e-8
    assert abs(m15.loc["LOBO", "R2"] - 0.6941536904) < 1e-8

    sdgrain = pd.read_csv(RESULTS / "sdgrain_m15_validation.csv")
    sdgrain_lobo = sdgrain[sdgrain["protocol"] == "LOBO"].set_index("model")
    assert sdgrain_lobo.loc["M3 + additive SD_grain", "R2"] < 0.60
    assert sdgrain_lobo.loc["M15: M3 + SD_grain + SD_grain*d^-1/2", "R2"] > 0.69

    feature_audit = pd.read_csv(RESULTS / "sdgrain_feature_audit.csv")
    campaign_auc = feature_audit[feature_audit["audit"] == "campaign separation"].set_index("target_or_term")
    assert campaign_auc.loc["HoldTime", "value"] == 1.0
    assert campaign_auc.loc["dH_mix/HoldTime^2", "value"] > 0.96
    assert campaign_auc.loc["(6.93-d)/SD_grain", "value"] < 0.54

    manuscript = (ROOT / "paper" / "main.tex").read_text()
    supplement = (ROOT / "paper" / "supplementary.tex").read_text()
    assert "222.3+84.5" in manuscript
    assert "post-selection fixed-form" in manuscript
    assert "No deployment claim" in manuscript
    combined_text = (manuscript + supplement).lower()
    for forbidden in (
        "campaign-out",
        "campaign holdout",
        "leave-one-campaign",
        "campaign $r^2$",
    ):
        assert forbidden not in combined_text
    assert "external literature" in combined_text
    assert "singularity audit" in combined_text

    family_definitions = [
        r"\newcommand{\familyone}{Classical Hall-Petch}",
        r"\newcommand{\familytwo}{Physics-derived descriptors}",
        r"\newcommand{\familythree}{Composition/processing}",
        r"\newcommand{\familyfour}{Non-linear ML (ARMOTE-CV)}",
        r"\newcommand{\familyfive}{Symbolic regression}",
    ]
    family_headings = [
        r"\subsection{Family 1: \familyone}",
        r"\subsection{Family 2: \familytwo}",
        r"\subsection{Family 3: \familythree}",
        r"\subsection{Family 4: \familyfour}",
        r"\subsection{Family 5: \familyfive}",
    ]
    supplement_headings = [heading.replace("subsection", "section") for heading in family_headings]

    for definition in family_definitions:
        assert definition in manuscript
        assert definition in supplement

    methods = manuscript.split(r"\section{Methods}", 1)[1].split(
        r"\subsection{Validation hierarchy and score uncertainty}", 1
    )[0]
    results = manuscript.split(r"\section{Results}", 1)[1].split(
        r"\subsection{Hardness--yield correspondence}", 1
    )[0]
    assert_in_order(methods, family_headings)
    assert_in_order(results, family_headings)
    assert_in_order(supplement, supplement_headings)

    assert "Tier~1" not in manuscript + supplement
    assert "descriptor-rich tiers" not in manuscript + supplement
    assert "PySR used F1" not in supplement
    print("Revision validation passed.")


if __name__ == "__main__":
    main()
