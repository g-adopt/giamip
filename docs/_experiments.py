"""Build the GIAMIP experiment tables of the documentation pages.

Three pages show the GIAMIP experiments: the protocol page, the collection page
and the results page. All three build their table here, from the case files in
`giamip/resources/cases/giamip/`. No page types an experiment value, so a page
and a case file cannot disagree.

Each case file holds the protocol values and, in a `gadopt` block, the values
that the G-ADOPT team chose for the fields the protocol leaves open. A table
cell shows the G-ADOPT value in red where one exists, and the protocol value
otherwise.

This file is not a page. Quarto renders only the files that `_quarto.yml`
lists, and it lists no `.py` file in this directory.
"""

from __future__ import annotations

from giamip import cases

# The Markdown span that colours a G-ADOPT choice. Pandoc turns it into a
# styled HTML span. It is kept in this one place so that tests can require
# that no page types it directly.
RED = '[{}]{{style="color: red;"}}'

# Display labels for the protocol page. The case files store short ids; the
# protocol table uses the names below. A viscosity or elastic structure id
# that is not here is shown by the `name` of its model file.
RHEOLOGY_LABELS = {
    "open": "Open",
    "maxwell": "Maxwell",
    "burgers": "Burgers",
    "power-law": "Power-law",
    "burgers-power-law": "Burgers + power-law",
    "same as Exp02": "Same as Exp02",
}
TOPOGRAPHY_LABELS = {
    "open": "Open",
    "bedmap2": "Bedmap2 (ANT) + BedMachineV5 (Greenland)",
    "bedmap3": "Bedmap3 (ANT) + BedMachineV5 (Greenland)",
    "bedmach3": "BedMachineV3 (ANT) + BedMachineV5 (Greenland)",
}
COMPRESSIBILITY_LABELS = {
    "open": "Open",
    "compressible": "Compressible",
    "incompressible": "Incompressible",
}
TIER_LABELS = {1: "Tier 1 Experiments (Mandatory)",
               2: "Tier 2 Experiments (Optional)"}


def _label(field: str, value: str) -> str:
    """The protocol-page label of one value of one of `cases.GADOPT_FIELDS`."""
    if field == "rheology":
        return RHEOLOGY_LABELS[value]
    if field == "topography":
        return TOPOGRAPHY_LABELS[value]
    if field == "compressibility":
        return COMPRESSIBILITY_LABELS[value]
    if value == "open":
        return "Open"
    # A viscosity or an elastic structure: the name in its model file.
    return cases.earth_model(value)["name"]


def _cells(c: dict, labelled: bool) -> list[str]:
    """The five cells of `cases.GADOPT_FIELDS` for one case, in its order.

    Parameters
    ----------
    c
        A GIAMIP case.
    labelled
        True gives the display names of the protocol page. False gives the
        ids that the case files store, for the collection and results pages.
    """
    chosen = cases.gadopt_choices(c)
    out = []
    for field in cases.GADOPT_FIELDS:
        value = chosen.get(field, cases.protocol_value(c, field))
        text = _label(field, value) if labelled else value
        # Markdown tables need a "+" escaped only at the start of a cell, but
        # escaping it everywhere is harmless and keeps the rule simple.
        text = text.replace("+", "\\+")
        out.append(RED.format(text) if field in chosen else text)
    return out


def protocol_table() -> str:
    """The experiment table of the protocol page, in the protocol's layout."""
    rows = ["|  | Experiment ID | Global input ice model | Viscosity structure | "
            "Elastic structure | Earth rheology | Initial polar topography "
            "(GEBCO 2024 for rest of the globe) | Compressibility | Period |",
            "| ----- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    seen_tiers = set()
    for c in cases.cases("giamip"):
        # The tier label appears once, on the first row of its tier.
        tier = "" if c["tier"] in seen_tiers else TIER_LABELS[c["tier"]]
        seen_tiers.add(c["tier"])
        p = c["period"]
        period = (f"{p['start_ka_before_2000ce']} ka-"
                  f"{p['end_years_after_2000ce']} yrs into future")
        ice = c["forcing"]["ice_history"].replace("_", "\\_")
        rows.append("| " + " | ".join(
            [tier, c["experiment_id"], ice, *_cells(c, labelled=True), period]) + " |")
    return "\n".join(rows)


def id_table(link: bool, open_count: bool) -> str:
    """The experiment table of the collection and results pages, with ids.

    Parameters
    ----------
    link
        True links each experiment to its full specification on the results
        page. The collection page links; the results page is that target.
    open_count
        True adds a column with the number of fields the protocol leaves open.
    """
    head = ("| Experiment | Tier | Ice history | Viscosity | Elastic | Rheology | Topography "
            "| Compressibility | Start |")
    rule = "| --- | ---: | --- | --- | --- | --- | --- | --- | ---: |"
    if open_count:
        head += " Open choices |"
        rule += " ---: |"
    rows = [head, rule]
    for c in cases.cases("giamip"):
        eid = f"`{c['experiment_id']}`"
        if link:
            eid = f"[{eid}](../results/index.qmd#{c['id']})"
        cells = [eid, str(c["tier"]), c["forcing"]["ice_history"],
                 *_cells(c, labelled=False),
                 f"{c['period']['start_ka_before_2000ce']} ka"]
        if open_count:
            cells.append(str(len(c["open_choices"])))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
