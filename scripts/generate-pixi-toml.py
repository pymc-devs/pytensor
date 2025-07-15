#!/usr/bin/env python3

"""
Generate a `pixi.toml` file.

Overview:

To a very good approximation, this script just runs RAW_COMMAND,
and writes the output to `pixi.toml`.

This script is necessary in order to do some minor pre- and post-processing.

Context:

Pixi is a modern project-centric manager of Conda environments.
It runs completely independently of more traditional tools like `conda`
or `mamba`, or non-Conda-enabled Python environment managers like
`pip` or `uv`. As such, we need to define the dependencies and Conda
environments that pixi should automatically manage. This is specified
in a `pixi.toml` file.

The sources of truth for the dependencies in this project are:
- `environment.yml`
- `pyproject.toml`

The goal of this script is to maintain the original sources of truth and
in a completely automatic way generate a valid `pixi.toml` from them.

Currently, pixi has no mechanism for importing `environment.yml` files,
and when loading `pyproject.toml` files it does not attempt to convert
the PyPI dependencies into conda-forge dependencies. On the other hand,
`conda-lock` does both of these things, and recently added support for
generating `pixi.toml` files from `environment.yml` and `pyproject.toml`
files.

PyTensor has some rather complicated dependencies, and so we do some
pre- and post-processing in this script in order to work around some
idiosyncrasies.

Technical details:

This script creates a temporary working directory `pixi-working/` in
the project root, copies pre-processed versions of `environment.yml`,
`pyproject.toml`, and `scripts/environment-blas.yml` into it, and then
runs `conda-lock` via `pixi exec` in order to generate the contents of
a `pixi.toml` file.

The generated `pixi.toml` contents are then post-processed and written
to the project root.

The pre-processing steps are:

- Remove the `complete` and `development` optional dependencies from
  `pyproject.toml`.
- Remove the BLAS dependencies from `environment.yml`. (They are redefined
  in `scripts/environment-blas.yml` on a platform-by-platform basis to
  work around limitations in `conda-lock`'s selector parsing.)

The post-processing steps are:

- Add an explanatory header comment to the `pixi.toml` file.
- Remove the `win-64` platform from the jax feature (jaxlib is not
  available for Windows from conda-forge).
- Add the default solve group to each environment entry.
- Comment out the non-default environments.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import tomlkit
import tomlkit.items


class WorkingDirectoryPaths(NamedTuple):
    """Paths within the temporary working directory"""

    working_path: Path
    working_pyproject_file: Path
    working_environment_file: Path
    working_environment_blas_file: Path
    pixi_toml_raw_file: Path
    gitignore_file: Path


RAW_COMMAND = """
pixi exec conda-lock render-lock-spec \
    --channel=conda-forge \
    --kind=pixi.toml \
    --file=environment.yml \
    --file=scripts/environment-blas.yml \
    --file=pyproject.toml \
    --stdout \
    --pixi-project-name=pytensor \
    --editable=pytensor=.
"""

PARSED_COMMAND = shlex.split(RAW_COMMAND)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pixi.toml from environment files"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that pixi.toml is consistent with environment files, don't write",
    )
    args = parser.parse_args()

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

    environment_blas_file = project_root / "scripts" / "environment-blas.yml"
    working_environment_blas_file = working_path / "scripts" / "environment-blas.yml"
    # This is for our exclusive use, so it doesn't need to be preprocessed.
    working_environment_blas_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(environment_blas_file, working_environment_blas_file)

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

    working_directory_paths = WorkingDirectoryPaths(
        working_path=working_path,
        working_pyproject_file=working_pyproject_file,
        working_environment_file=working_environment_file,
        working_environment_blas_file=working_environment_blas_file,
        pixi_toml_raw_file=pixi_toml_raw_file,
        gitignore_file=gitignore_file,
    )

    # Generate the pixi.toml content
    pixi_toml_content = tomlkit.dumps(pixi_toml_data)

    if args.verify_only:
        # Compare with existing pixi.toml
        pixi_toml_file = project_root / "pixi.toml"
        if not pixi_toml_file.exists():
            print("ERROR: pixi.toml does not exist")  # noqa: T201
            cleanup_working_directory(working_directory_paths)
            sys.exit(1)

        existing_content = pixi_toml_file.read_text()
        if existing_content != pixi_toml_content:
            print("ERROR: pixi.toml is not consistent with environment files")  # noqa: T201
            print("Run 'python scripts/generate-pixi-toml.py' to update it")  # noqa: T201
            cleanup_working_directory(working_directory_paths)
            sys.exit(1)

        print("SUCCESS: pixi.toml is consistent with environment files")  # noqa: T201
        cleanup_working_directory(working_directory_paths)
        sys.exit(0)
    else:
        # Write the pixi.toml file to the project root
        (project_root / "pixi.toml").write_text(pixi_toml_content)
        cleanup_working_directory(working_directory_paths)


def cleanup_working_directory(working_paths: WorkingDirectoryPaths):
    """Clean up the temporary working directory and files"""
    working_paths.working_pyproject_file.unlink()
    working_paths.working_environment_file.unlink()
    working_paths.working_environment_blas_file.unlink()
    working_paths.working_environment_blas_file.parent.rmdir()
    working_paths.pixi_toml_raw_file.unlink()
    working_paths.gitignore_file.unlink()
    working_paths.working_path.rmdir()


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
    pixi_toml_data = use_only_default_solve_group(pixi_toml_data)
    pixi_toml_data = comment_out_environments_that_wont_solve(pixi_toml_data)
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


def comment_out_environments_that_wont_solve(
    pixi_toml_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    """
    Comment out the lines defining environments that won't solve.

    Reference: <https://github.com/prefix-dev/pixi/issues/2725>

    >>> input_data = tomlkit.loads('''
    ... [environments]
    ... tests = {features = ["tests"]}
    ... dev = {features = ["dev"]}
    ... ''')
    >>> output_data = comment_out_environments_that_wont_solve(input_data)
    >>> print(tomlkit.dumps(output_data))
    [environments]
    dev = {features = ["dev"]}
    # tests = {features = ["tests"]}
    """
    environments_item = pixi_toml_data.get("environments")
    assert isinstance(environments_item, tomlkit.items.Table)

    # Create a new table to rebuild with the desired order
    new_environments_table = tomlkit.table()

    COMMENT_TEXT_1 = (
        "Disable non-default environments pending "
        "<https://github.com/prefix-dev/pixi/issues/2725>"
    )
    COMMENT_TEXT_2 = (
        "If you want to re-enable some of these, it's recommended to "
        "remove the solve-group key."
    )

    # Process each key-value pair in the original table
    for key, value in environments_item.items():
        if key == "default":
            # Keep the default environment as-is
            new_environments_table.append(key, value)

            # Add explanatory comment at the beginning
            new_environments_table.append(None, tomlkit.nl())
            new_environments_table.append(None, tomlkit.comment(COMMENT_TEXT_1))
            new_environments_table.append(None, tomlkit.comment(COMMENT_TEXT_2))
        else:
            # Convert other environments to comments
            comment_text = f"{key} = {value.as_string()}"

            # Note: We don't preserve original comments here since they
            # don't make sense in the commented-out context
            new_environments_table.append(None, tomlkit.comment(comment_text))

    # Replace the original environments table with the new one
    pixi_toml_data["environments"] = new_environments_table

    return pixi_toml_data


def use_only_default_solve_group(
    pixi_toml_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    """
    Use only the default solve group.

    TODO: upstream this to conda-lock.
    """
    pixi_toml_data = _convert_environment_items_to_dict_form(pixi_toml_data)
    pixi_toml_data = _add_default_solve_group(pixi_toml_data)
    return pixi_toml_data


def _convert_environment_items_to_dict_form(
    pixi_toml_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    """
    Convert the environment items to dict form.

    >>> input_data = tomlkit.loads('''
    ... [environments]
    ... tests = ["tests"]
    ... dev = {features = ["dev"]}
    ... ''')
    >>> output_data = _convert_environment_items_to_dict_form(input_data)
    >>> print(tomlkit.dumps(output_data))
    [environments]
    tests = {features = ["tests"]}
    dev = {features = ["dev"]}
    """
    environments_item = pixi_toml_data.get("environments")
    assert isinstance(environments_item, tomlkit.items.Table)

    # Modify the table in-place to preserve comments and formatting
    for key, value in list(environments_item.items()):
        if isinstance(value, tomlkit.items.Array):
            # Create an inline table and try to preserve comments
            inline_table = tomlkit.inline_table()
            inline_table["features"] = value.unwrap()

            # Try to preserve any comment from the original array
            if hasattr(value, "trivia") and value.trivia.comment:
                inline_table.comment(value.trivia.comment)

            environments_item[key] = inline_table

    return pixi_toml_data


def _add_default_solve_group(
    pixi_toml_data: tomlkit.TOMLDocument,
) -> tomlkit.TOMLDocument:
    """
    Add the default solve group to each environment entry.

    >>> input_data = tomlkit.loads('''
    ... [environments]
    ... tests = {features = ["tests"]}
    ... dev = {features = ["dev"]}
    ... ''')
    >>> output_data = _add_default_solve_group(input_data)
    >>> print(tomlkit.dumps(output_data))
    [environments]
    tests = {features = ["tests"], solve-group = "default"}
    dev = {features = ["dev"], solve-group = "default"}
    """
    environments_item = pixi_toml_data.get("environments")
    assert isinstance(environments_item, tomlkit.items.Table)

    for value in environments_item.values():
        if isinstance(value, tomlkit.items.InlineTable):
            if "solve-group" not in value:
                value["solve-group"] = "default"

    return pixi_toml_data


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
