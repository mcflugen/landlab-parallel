from __future__ import annotations

import os

import nox

ROOT = os.path.dirname(os.path.abspath(__file__))


@nox.session
def build(session: nox.Session) -> None:
    """Build sdist and wheel dists."""
    outdir = os.path.join(ROOT, "dist")
    session.install("build")
    session.run("python", "-m", "build", "--outdir", outdir)
    session.log(f"Built distributions can be found in {outdir}")
    return outdir


@nox.session
def install(session: nox.Session) -> None:
    first_arg = session.posargs[0] if session.posargs else None

    if first_arg:
        if os.path.isfile(first_arg):
            session.install(first_arg)
        elif os.path.isdir(first_arg):
            session.install(
                "landlab-parallel",
                f"--find-links={first_arg}",
                "--no-deps",
                "--no-index",
            )
        else:
            session.error("path must be a source distribution or folder")
    else:
        session.install("-e", ".")


@nox.session
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.install("pytest")
    install(session)

    session.run("pytest", "--doctest-modules", "--pyargs", "landlab_parallel", "tests/")


@nox.session
def coverage(session: nox.Session) -> None:
    """Run coverage"""
    session.install("coverage", "pytest")
    session.install("-e", ".")

    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "src/landlab_parallel",
        "tests",
        "--doctest-modules",
        env={"COVERAGE_CORE": "sysmon"},
    )

    session.run("coverage", "report", "--ignore-errors", "--show-missing")
    session.run("coverage", "xml", "-o", "coverage.xml")


@nox.session
def lint(session: nox.Session) -> None:
    """Look for lint."""
    skip_hooks = ("file-contents-sorter",)
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", env={"SKIP": ",".join(skip_hooks)})
