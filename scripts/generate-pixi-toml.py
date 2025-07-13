#!/usr/bin/env python3

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import tomlkit
import tomlkit.items


RAW_COMMAND = """
pixi exec conda-lock render-lock-spec \
    --channel=conda-forge \
    --kind=pixi.toml \
    --file=environment.yml \
    --file=pyproject.toml \
    --stdout \
    --pixi-project-name=pytensor \
    --editable=pytensor=.
"""

PARSED_COMMAND = shlex.split(RAW_COMMAND)


def main():
    # Check if pixi is installed
    if not shutil.which("pixi"):
        print("pixi is not installed. See <https://pixi.sh/latest/#installation>")  # noqa: T201
        sys.exit(1)

    current_dir = Path(__file__).parent
    assert current_dir.name == "scripts"
    project_root = current_dir.parent

    pyproject_file = project_root / "pyproject.toml"

    pyproject_data = tomlkit.loads(pyproject_file.read_text())

    pyproject_data = preprocess_pyproject_data(pyproject_data)

    # Make temporary working directory
    working_path = project_root / "pixi-working"
    working_path.mkdir(parents=True, exist_ok=True)
    gitignore_file = working_path / ".gitignore"
    gitignore_file.write_text("*")

    working_pyproject_file = working_path / "pyproject.toml"
    with working_pyproject_file.open("w") as fh:
        tomlkit.dump(pyproject_data, fh)

    raw_environment_file = project_root / "environment.yml"
    raw_environment_data = raw_environment_file.read_text()

    environment_data = preprocess_environment_data(raw_environment_data)

    working_environment_file = working_path / "environment.yml"
    with working_environment_file.open("w") as fh:
        fh.write(environment_data)

    print(f"Running the command:\n{shlex.join(PARSED_COMMAND)}\n")  # noqa: T201
    result = subprocess.run(
        PARSED_COMMAND, check=True, capture_output=True, cwd=working_path
    )

    warnings = result.stderr.decode("utf-8")
    if warnings:
        print(f"Warnings: \n{warnings}\n")  # noqa: T201

    pixi_toml_raw_data = tomlkit.loads(result.stdout.decode("utf-8"))

    pixi_toml_raw_file = working_path / "pixi.toml"
    pixi_toml_raw_file.write_text(tomlkit.dumps(pixi_toml_raw_data))

    pixi_toml_data = postprocess_pixi_toml_data(pixi_toml_raw_data)

    # Write the pixi.toml file to the project root
    (project_root / "pixi.toml").write_text(tomlkit.dumps(pixi_toml_data))

    # Clean up
    working_pyproject_file.unlink()
    working_environment_file.unlink()
    pixi_toml_raw_file.unlink()
    gitignore_file.unlink()
    working_path.rmdir()


def preprocess_pyproject_data(
    pyproject_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    pyproject_data = remove_extraneous_features(pyproject_data)
    return pyproject_data


def remove_extraneous_features(
    pyproject_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    """
    Remove the `complete` and `development` optional dependencies from pyproject.toml.

    >>> input_data = tomlkit.loads('''
    ... [project.optional-dependencies]
    ... complete = ["pytensor[jax]", "pytensor[numba]"]
    ... development = ["pytensor[complete]", "pytensor[tests]", "pytensor[rtd]"]
    ... rtd = ["sphinx>=5.1.0,<6", "pygments", "pydot"]
    ... ''')
    >>> output_data = remove_extraneous_features(input_data)
    >>> print(tomlkit.dumps(output_data))
    [project.optional-dependencies]
    rtd = ["sphinx>=5.1.0,<6", "pygments", "pydot"]
    """
    project_item = pyproject_data.get("project")
    assert isinstance(project_item, tomlkit.items.Table)
    optional_dependencies_item = project_item.get("optional-dependencies")
    assert isinstance(optional_dependencies_item, tomlkit.items.Table)

    del optional_dependencies_item["complete"]
    del optional_dependencies_item["development"]

    return pyproject_data


def postprocess_pixi_toml_data(
    pixi_toml_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    pixi_toml_data = restrict_jax_platforms(pixi_toml_data)
    pixi_toml_data = add_header_comment(pixi_toml_data)
    return pixi_toml_data


def restrict_jax_platforms(
    pixi_toml_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    """
    Remove the `win-64` platform from the jax feature.

    Jax isn't available for Windows from conda-forge, so it needs to be
    explicitly excluded, otherwise pixi will not be able to solve the
    environment.

    Specifically, above the `[feature.jax.dependencies]` table, create a
    `[feature.jax]` table with a `platforms` key containing the values from
    `[project.platforms]` but with `win-64` removed.

    >>> input_data = tomlkit.loads('''
    ... [project]
    ... platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]
    ... [feature.jax.dependencies]
    ... jax = "*"
    ... ''')
    >>> output_data = restrict_jax_platforms(input_data)
    >>> print(tomlkit.dumps(output_data))
    [project]
    platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]
    [feature.jax]
    platforms = ["linux-64", "osx-64", "osx-arm64"]
    [feature.jax.dependencies]
    jax = "*"
    """
    # Get the platforms from the project section - unwrap to get the actual list
    project_item = pixi_toml_data["project"]
    assert isinstance(project_item, tomlkit.items.Table)

    platforms_item = project_item["platforms"]
    assert isinstance(platforms_item, tomlkit.items.Array)
    project_platforms = platforms_item.unwrap()

    # Filter out win-64
    jax_platforms = [platform for platform in project_platforms if platform != "win-64"]

    # Access the feature section and modify the jax subsection
    feature_item = pixi_toml_data.get("feature")
    assert isinstance(feature_item, tomlkit.items.Table)
    jax_item = feature_item.get("jax")
    assert isinstance(jax_item, tomlkit.items.Table)

    # Create a new jax feature table with platforms first, then dependencies
    new_jax_feature = tomlkit.table()
    new_jax_feature.add("platforms", jax_platforms)

    # Add all existing keys from the jax feature (like dependencies)
    for key, value in jax_item.items():
        new_jax_feature.add(key, value)

    # Replace the jax feature with the new one
    feature_item["jax"] = new_jax_feature

    return pixi_toml_data


def add_header_comment(pixi_toml_data: tomlkit.TOMLDocument) -> tomlkit.TOMLDocument:
    """
    Add a header comment to the pixi.toml file.
    """
    header_lines = [
        "THIS FILE WAS GENERATED BY `scripts/generate-pixi-toml.py`",
        "You can edit this file locally, but if you want to push changes",
        "upstream to pymc-devs/pytensor, then you should instead modify",
        "the script.",
    ]

    # Create a new document with the header comment
    new_doc = tomlkit.document()

    # Add header comments
    for line in header_lines:
        new_doc.add(tomlkit.comment(line))

    # Add a blank line after the header
    new_doc.add(tomlkit.nl())

    # Add all the existing content from the original document
    for key, value in pixi_toml_data.items():
        new_doc.add(key, value)

    return new_doc


def preprocess_environment_data(
    raw_environment_data: str,
) -> str:
    environment_data = remove_blas_dependencies(raw_environment_data)
    return environment_data


def remove_blas_dependencies(
    raw_environment_data: str,
) -> str:
    """
    Remove the BLAS dependencies from the environment.yml file.
    """
    lines = raw_environment_data.splitlines()

    def filter(line: str) -> bool:
        TO_REMOVE = [
            "- mkl",
            "- mkl-service",
            "- libblas=*=*mkl",
        ]
        return line.strip() not in TO_REMOVE

    lines = [line for line in lines if filter(line)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
