""""    
    Copyright (C) 2026  Digiser

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import Any

from .models import MaterialProfile


def extract_recyclate_content(smc_payload: dict[str, Any] | None) -> float | None:
    """
    Extract a recyclate-content value from the secondary material content payload.

    Args:
            smc_payload (dict[str, Any] | None): The smc payload used by this method.

    Steps:
        1. Read the inputs required to extract recyclate content.
        2. Perform the operations needed to extract recyclate content.
        3. Return the resulting value or update the relevant application state.

    Returns:
            float | None: The extracted recyclate-content value.
    """
    if not smc_payload:
        return None

    # The SMC shared model exposes a list of materials with details on primary and secondary shares.
    # Because providers may serialize slightly different JSON envelopes, keep the extraction tolerant.
    materials = (
        smc_payload.get("secondaryMaterialContent")
        or smc_payload.get("secondaryMaterialContentMaterials")
        or []
    )
    if isinstance(materials, dict):
        materials = [materials]
    if not isinstance(materials, list):
        return None

    total_percentage = 0.0
    found = False
    for item in materials:
        if not isinstance(item, dict):
            continue
        for key in (
            "secondaryMaterialContentPercentage",
            "secondarySharePercentage",
            "recycledContentPercentage",
            "percentage",
        ):
            value = item.get(key)
            if isinstance(value, (int, float)):
                total_percentage += float(value)
                found = True
                break
    return round(total_percentage, 4) if found else None


def extract_cmp_flags(cmp_payload: dict[str, Any] | None) -> dict[str, bool | None]:
    """
    Derive harmfulness flags from the Chemical Material Passport payload.

    Args:
            cmp_payload (dict[str, Any] | None): The cmp payload used by this method.

    Steps:
        1. Read the inputs required to extract cmp flags.
        2. Perform the operations needed to extract cmp flags.
        3. Return the resulting value or update the relevant application state.

    Returns:
            dict[str, bool | None]: The derived harmfulness flags.
    """
    if not cmp_payload:
        return {
            "environmental_harmfulness": None,
            "health_harmfulness": None,
        }

    substances_of_concern = []
    soc = cmp_payload.get("substanceOfConcern")
    if isinstance(soc, dict):
        substances_of_concern = soc.get("substanceOfConcern", []) if isinstance(soc.get("substanceOfConcern"), list) else []
    elif isinstance(soc, list):
        substances_of_concern = soc

    health = False
    environment = False

    for substance in substances_of_concern:
        text = str(substance).lower()
        if any(token in text for token in ["carcin", "mutagen", "reprotox", "acute tox", "skin corr", "stot"]):
            health = True
        if any(token in text for token in ["aquatic", "ozone", "environment", "long lasting effects"]):
            environment = True

    return {
        "environmental_harmfulness": environment,
        "health_harmfulness": health,
    }


def to_custom_output(profile: MaterialProfile, material_key: str) -> dict[str, Any]:
    """
    Convert a resolved material profile into the target materials JSON structure.

    Args:
            profile (MaterialProfile): The resolved material profile to convert or serialize.
            material_key (str): The optional key to use as the top-level material entry name in the output JSON.

    Steps:
        1. Read the resolved values from the material profile.
        2. Map those values into the target materials JSON structure.
        3. Return the generated output payload.

    Returns:
            dict: The output payload formatted for the target materials database structure.
    """
    cmp_flags = extract_cmp_flags(profile.cmp_payload)
    return {
        "materials": {
            material_key: {
                "bpnl": profile.supplier_bpnl,
                "bpnl_source": profile.supplier_bpnl_source,
                "bpns": profile.supplier_site_bpns,
                "bpna": profile.supplier_address_bpna,
                "manufacturer_part_id": profile.manufacturer_part_id,
                "part_instance_id": profile.part_instance_id,
                "lifecycle": profile.lifecycle,
                "material_name": profile.material_name,
                "candidate_site_bpnss": profile.candidate_site_bpnss,
                "recycling_percentage": None,
                "recyclate_content": extract_recyclate_content(profile.smc_payload),
                "recycling_criticality": None,
                "incineration_for_energy_recovery": None,
                "environmental_harmfulness": cmp_flags["environmental_harmfulness"],
                "health_harmfulness": cmp_flags["health_harmfulness"],
                "monomaterial": None,
                "additives_or_fillers": None,
                "surface_coatings": None,
                "availability": None,
                "density [kg/mm^3]": None,
                "CO2eq [kg]": None,
                "raw": {
                    "part_type_information": profile.part_type_information_payload,
                    "smc_semantic_id": profile.smc_semantic_id,
                    "smc": profile.smc_payload,
                    "chemical_material_passport": profile.cmp_payload,
                    "bpdm_site": profile.bpdm_site,
                    "bpdm_address": profile.bpdm_address,
                },
            }
        }
    }
