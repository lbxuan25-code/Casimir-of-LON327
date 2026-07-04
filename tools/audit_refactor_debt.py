"""Static refactor-debt audit helper.

This script intentionally avoids importing ``lno327``. It scans source text and
AST import nodes only, so it has no physics or package import side effects.
"""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD_MODULES = (
    "conductivity",
    "bdg_response",
    "nonlocal_response",
    "bdg_nonlocal_response",
    "finite_q_primitives",
    "tb_fourier",
    "ward_response",
    "ward_validation",
    "response_interface",
    "static_response",
    "response_conventions",
    "reflection_input",
    "casimir",
)

EXPENSIVE_PATTERNS = (
    r"np\.linalg\.eigh",
    r"diagonalize_hermitian",
    r"normal_eigensystem_from_model",
    r"bdg_eigensystem_from_model",
    r"fermi_function",
    r"negative_fermi_derivative",
    r"hopping_terms\(",
    r"normal_state_hopping_terms\(",
    r"LNO327FourOrbitalSpec\(",
    r"SymmetryBdG2BandSpec\(",
    r"PairingAmplitudes\(",
    r"spec\.normal_hamiltonian",
    r"spec\.bdg_hamiltonian",
    r"spec\.velocity_operator",
    r"spec\.mass_operator",
    r"spec\.peierls_hamiltonian_vector_vertex",
    r"spec\.peierls_hamiltonian_contact_vertex",
    r"transform_operator_to_band_basis",
)


def py_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        path = ROOT / root
        if path.exists():
            files.extend(sorted(path.rglob("*.py")))
    return files


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                imports.append("." * node.level + module)
            else:
                imports.append(module)
    return imports


def references_old_module(path: Path, module: str) -> bool:
    text = path.read_text()
    needles = (
        f"lno327.{module}",
        f"from .{module}",
        f"import .{module}",
    )
    return any(needle in text for needle in needles)


def category(path: Path) -> str:
    path_rel = rel(path)
    if path_rel.startswith("src/lno327/"):
        return "src"
    if path_rel.startswith("tests/"):
        return "tests"
    if path_rel.startswith("validation/"):
        return "validation"
    if path_rel.startswith("scripts/"):
        return "scripts"
    return "other"


def old_inventory() -> dict[str, dict[str, list[str]]]:
    files = py_files("src/lno327", "tests", "validation", "scripts")
    old_paths = {f"src/lno327/{module}.py" for module in OLD_MODULES}
    inventory: dict[str, dict[str, list[str]]] = {}
    for module in OLD_MODULES:
        buckets: dict[str, list[str]] = defaultdict(list)
        for path in files:
            path_rel = rel(path)
            if path_rel in old_paths:
                continue
            if references_old_module(path, module):
                buckets[category(path)].append(path_rel)
        inventory[module] = {name: values for name, values in buckets.items()}
    return inventory


def expensive_matches() -> dict[str, Counter[str]]:
    compiled = [(pattern, re.compile(pattern)) for pattern in EXPENSIVE_PATTERNS]
    matches: dict[str, Counter[str]] = {}
    for path in py_files("src/lno327", "tests", "validation", "scripts"):
        text = path.read_text()
        counts = Counter({pattern: len(regex.findall(text)) for pattern, regex in compiled})
        counts = Counter({key: value for key, value in counts.items() if value})
        if counts:
            matches[rel(path)] = counts
    return matches


def public_surface_counts() -> dict[str, int]:
    files = (
        "src/lno327/api.py",
        "src/lno327/__init__.py",
        "src/lno327/response/__init__.py",
        "src/lno327/electrodynamics/__init__.py",
        "src/lno327/collective/__init__.py",
        "src/lno327/casimir/__init__.py",
        "src/lno327/models/__init__.py",
    )
    counts: dict[str, int] = {}
    for file_name in files:
        path = ROOT / file_name
        if not path.exists():
            counts[file_name] = 0
            continue
        tree = ast.parse(path.read_text())
        exported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                    if isinstance(node.value, ast.List):
                        exported = [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
        counts[file_name] = len(exported)
    return counts


def file_counts() -> dict[str, int]:
    return {
        "src/lno327 python files": len(py_files("src/lno327")),
        "src/lno327 packages": len(list((ROOT / "src" / "lno327").rglob("__init__.py"))),
        "tests python files": len(py_files("tests")),
        "validation/scripts python files": len(py_files("validation/scripts")),
        "scripts python files": len(py_files("scripts")),
        "tracked validation output files": len([p for p in (ROOT / "validation" / "outputs").rglob("*") if p.is_file()])
        if (ROOT / "validation" / "outputs").exists()
        else 0,
    }


def print_table(title: str, headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    print(f"\n## {title}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def main() -> None:
    inventory = old_inventory()
    rows = []
    for module in OLD_MODULES:
        buckets = inventory[module]
        rows.append(
            (
                f"src/lno327/{module}.py",
                len(buckets.get("src", [])),
                len(buckets.get("tests", [])),
                len(buckets.get("validation", [])),
                len(buckets.get("scripts", [])),
            )
        )
    print("# Refactor Debt Static Audit Summary")
    print_table("Old module references", ("old_file", "src", "tests", "validation", "scripts"), rows)
    print_table("Public surface counts", ("file", "__all__ names"), list(public_surface_counts().items()))
    print_table("File counts", ("category", "count"), list(file_counts().items()))

    expensive = expensive_matches()
    top = sorted(
        ((path, sum(counts.values())) for path, counts in expensive.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:25]
    print_table("Top expensive-pattern files", ("file", "matches"), top)


if __name__ == "__main__":
    main()
