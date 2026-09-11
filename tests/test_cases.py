"""Every case is well formed and points at things that exist."""

import pytest

from gia_mip import cases

COLLECTIONS = {"spada2011", "martinec2018", "giamip"}


def test_the_expected_number_of_cases():
    """Ten Spada tests, five Martinec cases, twelve GIAMIP experiments."""
    assert len(cases.ids("spada2011")) == 10
    assert len(cases.ids("martinec2018")) == 5
    assert len(cases.ids("giamip")) == 12
    assert len(cases.ids()) == 27


def test_ids_are_unique():
    """_index raises on a duplicate, so building it at all is the check."""
    assert len(cases.ids()) == len(set(cases.ids()))


def test_every_case_declares_a_known_collection():
    for c in cases.cases():
        assert c["collection"] in COLLECTIONS, c["id"]


def test_every_case_cites_its_source():
    for c in cases.cases():
        assert c["source"].get("citation"), c["id"]


def test_every_benchmark_case_names_an_earth_model_that_exists():
    for c in cases.cases("spada2011") + cases.cases("martinec2018"):
        model = cases.earth_model(c["earth_model"])
        assert model["id"] == c["earth_model"]


def test_every_giamip_case_names_structures_that_exist():
    """`open` means the group chooses; anything else must be a real model."""
    for c in cases.cases("giamip"):
        for key in ("viscosity", "elastic_structure"):
            value = c["earth"][key]
            if value in ("open", "same as Exp02"):
                continue
            assert cases.earth_model(value)["id"] == value, (c["id"], key)


def test_every_benchmark_case_names_a_load_table_that_exists():
    for c in cases.cases("spada2011") + cases.cases("martinec2018"):
        collection = c["loads"].split("/")[-1].removesuffix(".json")
        assert cases.loads(collection)


def test_martinec_cases_cover_A_to_E_with_the_published_specification():
    """Table 4 of the paper, pinned."""
    expected = {
        "A": (0, "L0", "T0", "B0", 10),
        "B": (1, "L1", "T0", "B1", 10),
        "C": (1, "L2", "T1", "B2", 15),
        "D": (2, "L2", "T1", "B2", 15),
        "E": (2, "L2", "T1", "B2*", 15),
    }
    for c in cases.cases("martinec2018"):
        letter = c["id"].removeprefix("martinec2018-")
        s = c["specification"]
        assert (s["sea_level_equation"], s["ice_height_model"],
                s["time_scenario"], s["ocean_basin"],
                s["terminal_time_kyr"]) == expected[letter], letter


def test_giamip_cases_use_the_giamip_constants_and_benchmarks_do_not():
    """The densities differ between the two, which is why they are per case."""
    for c in cases.cases("giamip"):
        assert c["constants"]["ice_density_kg_m3"] == 917
        assert c["constants"]["ocean_water_density_kg_m3"] == 1025
    for collection in ("spada2011", "martinec2018"):
        table = cases.loads(collection)
        assert table["constants"]["ice_density_kg_m3"] == 931
        assert table["constants"]["water_density_kg_m3"] == 1000


def test_giamip_cases_are_on_the_prescribed_grid():
    from gia_mip import grids
    for c in cases.cases("giamip"):
        assert c["grid"]["nlat"] == grids.NLAT
        assert c["grid"]["nlon"] == grids.NLON


def test_the_five_tier_one_experiments_are_marked_mandatory():
    mandatory = {c["experiment_id"] for c in cases.cases("giamip") if c["mandatory"]}
    assert mandatory == {"Exp01", "Exp02", "Exp03", "Exp04", "Exp05"}


def test_the_two_shortened_experiments_record_that_no_forcing_file_exists():
    """Exp06 and Exp07 run over 80 ka and only 122 ka files were delivered."""
    for eid in ("giamip-exp06", "giamip-exp07"):
        c = cases.case(eid)
        assert "truncation_required" in c["forcing"], eid


def test_case_raises_helpfully_for_an_unknown_id():
    with pytest.raises(KeyError, match="Known ids"):
        cases.case("no-such-case")


def test_earth_model_raises_helpfully_for_an_unknown_id():
    with pytest.raises(KeyError, match="Known"):
        cases.earth_model("no-such-model")
