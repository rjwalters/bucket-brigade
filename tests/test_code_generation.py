"""
Test suite for code generation system.

Validates that JSON definitions generate correct Python and TypeScript code,
and that all three representations remain consistent.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

# Import generated Python modules
from bucket_brigade.agents.archetypes_generated import (
    ARCHETYPES as PY_ARCHETYPES,
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)
from bucket_brigade.envs.scenarios_generated import (
    SCENARIO_REGISTRY as PY_SCENARIOS,
    Scenario,
    default_scenario,
    easy_scenario,
    hard_scenario,
)

# Constants
ROOT_DIR = Path(__file__).parent.parent
DEFINITIONS_DIR = ROOT_DIR / "definitions"
ARCHETYPES_JSON = DEFINITIONS_DIR / "archetypes.json"
SCENARIOS_JSON = DEFINITIONS_DIR / "scenarios.json"

# Expected archetype parameter count
EXPECTED_PARAM_COUNT = 10

# Expected scenario parameters
REQUIRED_SCENARIO_PARAMS = [
    "beta",
    "kappa",
    "A",
    "L",
    "c",
    "rho_ignite",
    "N_min",
    "p_spark",
    "N_spark",
]


class TestJSONDefinitions:
    """Test JSON definition files are valid and well-formed."""

    def test_archetypes_json_exists(self):
        """Archetypes JSON file exists."""
        assert ARCHETYPES_JSON.exists(), f"Missing {ARCHETYPES_JSON}"

    def test_scenarios_json_exists(self):
        """Scenarios JSON file exists."""
        assert SCENARIOS_JSON.exists(), f"Missing {SCENARIOS_JSON}"

    def test_archetypes_json_valid(self):
        """Archetypes JSON is valid and parseable."""
        with open(ARCHETYPES_JSON) as f:
            data = json.load(f)

        assert "version" in data, "Missing version field"
        assert "archetypes" in data, "Missing archetypes field"
        assert isinstance(data["archetypes"], dict), "Archetypes must be dict"

    def test_scenarios_json_valid(self):
        """Scenarios JSON is valid and parseable."""
        with open(SCENARIOS_JSON) as f:
            data = json.load(f)

        assert "version" in data, "Missing version field"
        assert "scenarios" in data, "Missing scenarios field"
        assert isinstance(data["scenarios"], dict), "Scenarios must be dict"

    def test_all_archetypes_have_params(self):
        """All archetypes have params array."""
        with open(ARCHETYPES_JSON) as f:
            data = json.load(f)

        for name, spec in data["archetypes"].items():
            assert "params" in spec, f"Archetype {name} missing params"
            assert isinstance(spec["params"], list), f"{name} params must be list"
            assert (
                len(spec["params"]) == EXPECTED_PARAM_COUNT
            ), f"{name} should have {EXPECTED_PARAM_COUNT} params, got {len(spec['params'])}"

    def test_all_archetypes_have_descriptions(self):
        """All archetypes have descriptions."""
        with open(ARCHETYPES_JSON) as f:
            data = json.load(f)

        for name, spec in data["archetypes"].items():
            assert "description" in spec, f"Archetype {name} missing description"
            assert isinstance(spec["description"], str), f"{name} description must be string"
            assert len(spec["description"]) > 0, f"{name} description is empty"

    def test_all_scenarios_have_required_params(self):
        """All scenarios have required parameters."""
        with open(SCENARIOS_JSON) as f:
            data = json.load(f)

        for name, spec in data["scenarios"].items():
            for param in REQUIRED_SCENARIO_PARAMS:
                assert param in spec, f"Scenario {name} missing {param}"
                assert isinstance(
                    spec[param], (int, float)
                ), f"{name}.{param} must be numeric"

    def test_all_scenarios_have_descriptions(self):
        """All scenarios have descriptions."""
        with open(SCENARIOS_JSON) as f:
            data = json.load(f)

        for name, spec in data["scenarios"].items():
            assert "description" in spec, f"Scenario {name} missing description"
            assert isinstance(spec["description"], str), f"{name} description must be string"
            assert len(spec["description"]) > 0, f"{name} description is empty"


class TestPythonGeneration:
    """Test Python code generation from JSON."""

    def test_python_archetypes_match_json(self):
        """Generated Python archetypes match JSON definitions."""
        with open(ARCHETYPES_JSON) as f:
            json_data = json.load(f)

        for name, spec in json_data["archetypes"].items():
            # Check archetype exists in Python
            assert (
                name in PY_ARCHETYPES
            ), f"Archetype {name} not found in Python ARCHETYPES"

            # Check parameters match
            expected_params = np.array(spec["params"], dtype=np.float32)
            actual_params = PY_ARCHETYPES[name]

            np.testing.assert_array_almost_equal(
                actual_params,
                expected_params,
                decimal=6,
                err_msg=f"Archetype {name} parameters don't match",
            )

    def test_python_individual_archetype_constants(self):
        """Individual Python archetype constants match JSON."""
        with open(ARCHETYPES_JSON) as f:
            json_data = json.load(f)

        # Map JSON names to Python constant names
        archetype_constants = {
            "firefighter": FIREFIGHTER_PARAMS,
            "free_rider": FREE_RIDER_PARAMS,
            "hero": HERO_PARAMS,
            "coordinator": COORDINATOR_PARAMS,
            "liar": LIAR_PARAMS,
        }

        for name, expected_params in archetype_constants.items():
            json_params = np.array(json_data["archetypes"][name]["params"], dtype=np.float32)
            np.testing.assert_array_almost_equal(
                expected_params,
                json_params,
                decimal=6,
                err_msg=f"Constant {name.upper()}_PARAMS doesn't match JSON",
            )

    def test_python_scenarios_match_json(self):
        """Generated Python scenarios match JSON definitions."""
        with open(SCENARIOS_JSON) as f:
            json_data = json.load(f)

        # Test scenario registry
        for name, spec in json_data["scenarios"].items():
            assert (
                name in PY_SCENARIOS
            ), f"Scenario {name} not found in Python SCENARIO_REGISTRY"

            # Get factory function
            factory = PY_SCENARIOS[name]
            scenario = factory(num_agents=10)

            # Verify it's a Scenario instance
            assert isinstance(scenario, Scenario), f"{name} factory didn't return Scenario"

            # Check each parameter matches JSON
            for param in REQUIRED_SCENARIO_PARAMS:
                actual_value = getattr(scenario, param)
                expected_value = spec[param]

                assert (
                    actual_value == expected_value
                ), f"Scenario {name}.{param}: expected {expected_value}, got {actual_value}"

    def test_python_scenario_factories(self):
        """Python scenario factories work correctly."""
        # Test that factories return valid scenarios
        scenario = default_scenario(num_agents=5)
        assert isinstance(scenario, Scenario)
        assert scenario.num_agents == 5

        scenario = easy_scenario(num_agents=10)
        assert isinstance(scenario, Scenario)
        assert scenario.num_agents == 10

        scenario = hard_scenario(num_agents=15)
        assert isinstance(scenario, Scenario)
        assert scenario.num_agents == 15


class TestTypeScriptGeneration:
    """Test TypeScript code generation from JSON."""

    def test_typescript_generator_runs(self):
        """TypeScript generator runs without errors."""
        result = subprocess.run(
            ["uv", "run", "python", "scripts/generate_typescript.py"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"TypeScript generator failed:\n{result.stderr}"

    def test_typescript_archetypes_file_exists(self):
        """Generated TypeScript archetypes file exists."""
        ts_file = ROOT_DIR / "web" / "src" / "data" / "archetypes.generated.ts"
        assert ts_file.exists(), f"Missing {ts_file}"

    def test_typescript_scenarios_file_exists(self):
        """Generated TypeScript scenarios file exists."""
        ts_file = ROOT_DIR / "web" / "src" / "utils" / "scenarioGenerator.generated.ts"
        assert ts_file.exists(), f"Missing {ts_file}"

    def test_typescript_archetypes_content(self):
        """Generated TypeScript archetypes contain expected content."""
        ts_file = ROOT_DIR / "web" / "src" / "data" / "archetypes.generated.ts"
        content = ts_file.read_text()

        # Load JSON for comparison
        with open(ARCHETYPES_JSON) as f:
            json_data = json.load(f)

        # Check all archetypes are present
        for name in json_data["archetypes"].keys():
            assert (
                f"{name}:" in content
            ), f"Archetype {name} not found in TypeScript file"

        # Check interfaces exist
        assert "export interface ArchetypeParams" in content
        assert "export interface Archetype" in content
        assert "export const ARCHETYPES" in content
        assert "export const PARAMETER_DESCRIPTIONS" in content

        # Check helper functions exist
        assert "export function getArchetype" in content
        assert "export function listArchetypes" in content

    def test_typescript_scenarios_content(self):
        """Generated TypeScript scenarios contain expected content."""
        ts_file = ROOT_DIR / "web" / "src" / "utils" / "scenarioGenerator.generated.ts"
        content = ts_file.read_text()

        # Load JSON for comparison
        with open(SCENARIOS_JSON) as f:
            json_data = json.load(f)

        # Check all scenarios are present
        for name in json_data["scenarios"].keys():
            constant_name = name.upper()
            assert (
                f"{constant_name}:" in content
            ), f"Scenario constant {constant_name} not found in TypeScript file"

        # Check types and constants exist
        assert "export const SCENARIO_TYPES" in content
        assert "export type ScenarioType" in content
        assert "const SCENARIO_TEMPLATES" in content

        # Check helper functions exist
        assert "export function generateScenario" in content
        assert "export function getScenarioTemplate" in content
        assert "export function getAllScenarioTypes" in content

    def test_typescript_has_proper_escaping(self):
        """TypeScript strings are properly escaped."""
        ts_archetype_file = ROOT_DIR / "web" / "src" / "data" / "archetypes.generated.ts"
        content = ts_archetype_file.read_text()

        # Check that apostrophes in strings are escaped
        # The description "Trust in others' signals" should be escaped
        assert "others\\' signals" in content or 'others" signals' not in content, (
            "Apostrophes in TypeScript strings must be properly escaped"
        )


class TestConsistency:
    """Test consistency between JSON, Python, and TypeScript."""

    def test_archetype_count_consistency(self):
        """Same number of archetypes in JSON and Python."""
        with open(ARCHETYPES_JSON) as f:
            json_data = json.load(f)

        json_count = len(json_data["archetypes"])
        python_count = len(PY_ARCHETYPES)

        assert json_count == python_count, (
            f"Archetype count mismatch: JSON has {json_count}, Python has {python_count}"
        )

    def test_scenario_count_consistency(self):
        """Same number of scenarios in JSON and Python."""
        with open(SCENARIOS_JSON) as f:
            json_data = json.load(f)

        json_count = len(json_data["scenarios"])
        python_count = len(PY_SCENARIOS)

        assert json_count == python_count, (
            f"Scenario count mismatch: JSON has {json_count}, Python has {python_count}"
        )

    def test_archetype_names_consistency(self):
        """Archetype names match between JSON and Python."""
        with open(ARCHETYPES_JSON) as f:
            json_data = json.load(f)

        json_names = set(json_data["archetypes"].keys())
        python_names = set(PY_ARCHETYPES.keys())

        assert json_names == python_names, (
            f"Archetype name mismatch:\n"
            f"  Only in JSON: {json_names - python_names}\n"
            f"  Only in Python: {python_names - json_names}"
        )

    def test_scenario_names_consistency(self):
        """Scenario names match between JSON and Python."""
        with open(SCENARIOS_JSON) as f:
            json_data = json.load(f)

        json_names = set(json_data["scenarios"].keys())
        python_names = set(PY_SCENARIOS.keys())

        assert json_names == python_names, (
            f"Scenario name mismatch:\n"
            f"  Only in JSON: {json_names - python_names}\n"
            f"  Only in Python: {python_names - json_names}"
        )


class TestCodeGenerationWarnings:
    """Test that generated files have appropriate warnings."""

    def test_python_archetypes_has_warning(self):
        """Generated Python archetypes file has DO NOT EDIT warning."""
        py_file = (
            ROOT_DIR / "bucket_brigade" / "agents" / "archetypes_generated.py"
        )
        content = py_file.read_text()

        assert "GENERATED FILE" in content or "DO NOT EDIT" in content, (
            "Generated Python file should have warning header"
        )

    def test_python_scenarios_has_warning(self):
        """Generated Python scenarios file has DO NOT EDIT warning."""
        py_file = ROOT_DIR / "bucket_brigade" / "envs" / "scenarios_generated.py"
        content = py_file.read_text()

        assert "GENERATED FILE" in content or "DO NOT EDIT" in content, (
            "Generated Python file should have warning header"
        )

    def test_typescript_archetypes_has_warning(self):
        """Generated TypeScript archetypes file has DO NOT EDIT warning."""
        ts_file = ROOT_DIR / "web" / "src" / "data" / "archetypes.generated.ts"
        content = ts_file.read_text()

        assert "GENERATED FILE" in content or "DO NOT EDIT MANUALLY" in content, (
            "Generated TypeScript file should have warning header"
        )

    def test_typescript_scenarios_has_warning(self):
        """Generated TypeScript scenarios file has DO NOT EDIT warning."""
        ts_file = ROOT_DIR / "web" / "src" / "utils" / "scenarioGenerator.generated.ts"
        content = ts_file.read_text()

        assert "GENERATED FILE" in content or "DO NOT EDIT MANUALLY" in content, (
            "Generated TypeScript file should have warning header"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
