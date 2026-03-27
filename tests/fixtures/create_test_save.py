"""Generate a minimal synthetic .sav fixture for testing."""

import zipfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

META = 'version="4.3.0.0"\nname="Test Galaxy"\ndate="2250.01.01"\n'

GAMESTATE = """version="4.3.0.0"
date="2250.01.01"
galaxy={ star_count=400 }
difficulty="grand_admiral"
player={ 0={ country=0 } }
country={
    0={
        name="United Nations of Earth"
        government={
            type="gov_democratic"
            authority="auth_democratic"
            civics={ "civic_beacon_of_liberty" "civic_idealistic_foundation" }
        }
        origin="origin_prosperous_unification"
        ethos={ ethic={ "ethic_fanatic_egalitarian" "ethic_xenophile" } }
        budget={
            current_month={
                income={ energy=142.50 minerals=98.30 alloys=43.00 }
                expenses={ energy=130.20 minerals=45.00 }
            }
        }
        tech_status={
            technology={
                0={ technology="tech_corvettes" level=1 }
                1={ technology="tech_destroyers" level=1 }
            }
        }
        owned_planets={ 0 4 }
        owned_systems={ 0 1 2 3 }
        fleet_power=5200.00
        naval_capacity={ used=52 max=120 }
        pop_count=45
        victory_score=85.0
        victory_rank=3
    }
    1={
        name="Tzynn Empire"
        government={ type="gov_dictatorial" authority="auth_dictatorial" }
        fleet_power=7800.00
        victory_score=120.0
        victory_rank=1
    }
}
galactic_object={
    0={
        name="Sol"
        star_class="sc_g"
        hyperlane={ 0={ to=1 } 1={ to=2 } }
        planet={ 0 1 2 3 }
    }
    1={
        name="Alpha Centauri"
        star_class="sc_k"
        hyperlane={ 0={ to=0 } 1={ to=3 } }
        planet={ 4 5 }
    }
}
planets={
    planet={
        0={
            name="Earth"
            planet_class="pc_continental"
            planet_size=25
            is_capital=yes
            districts={ district_city=3 district_mining=2 }
            buildings={ "building_capital_2" "building_alloy_foundry" }
            num_buildings_slots=8
            pop_count=30
            stability=55.0
            crime=5.0
            amenities=10.0
            amenities_usage=8.0
            total_housing=40.0
            free_housing=10.0
            colony_type="col_default"
        }
        4={
            name="Mars"
            planet_class="pc_arid"
            planet_size=16
            districts={ district_mining=3 }
            buildings={ "building_capital_1" }
            num_buildings_slots=4
            pop_count=12
            stability=48.0
            crime=2.0
            amenities=5.0
            amenities_usage=4.0
            total_housing=20.0
            free_housing=8.0
            colony_type="col_mining"
        }
    }
}
"""


def create_test_save() -> None:
    save_path = FIXTURES_DIR / "sample_save_small.sav"
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta", META)
        zf.writestr("gamestate", GAMESTATE)
    print(f"Created {save_path}")


if __name__ == "__main__":
    create_test_save()
