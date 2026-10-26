"""Check reviewed public files and built distributions before publication."""
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = frozenset((ROOT / "tools/public-files.txt").read_text().splitlines())
EGG_FILES = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt"}


def text_problem(data):
    text = data.decode("utf-8")
    if re.search(r"/Users/[A-Za-z0-9_.-]+/", text) or ("-----BEGIN " + "PRIVATE KEY-----") in text:
        return "private path or key marker"
    return None


def validate(name, data):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or name not in PUBLIC_FILES:
        return "outside reviewed public-file manifest"
    if path.suffix not in {".png", ".pdf"}:
        return text_problem(data)
    return None


def generated_source(name):
    if name in {"PKG-INFO", "setup.cfg"}:
        return True
    prefix = "src/ashrink.egg-info/"
    return name.startswith(prefix) and name[len(prefix):] in EGG_FILES


def check_archives(directory):
    archives = sorted(Path(directory).iterdir())
    wheels = [p for p in archives if p.suffix == ".whl"]
    sources = [p for p in archives if p.name.endswith(".tar.gz")]
    if len(archives) != 2 or len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("Expected exactly one wheel and one source archive.")
    runtime = {p.removeprefix("src/") for p in PUBLIC_FILES if p.startswith("src/ashrink/")}
    with zipfile.ZipFile(wheels[0]) as archive:
        members = archive.infolist()
        distinfo = {i.filename.split('/')[0] for i in members if '.dist-info/' in i.filename}
        if len(distinfo) != 1:
            raise SystemExit("Expected one wheel metadata directory.")
        prefix = next(iter(distinfo)) + "/"
        metadata = {prefix + p for p in {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt", "licenses/LICENSE", "licenses/NOTICE.md"}}
        names = {i.filename for i in members if not i.is_dir()}
        if names != runtime | metadata:
            raise SystemExit("Wheel contents differ from the reviewed runtime/metadata manifest.")
        for item in members:
            if stat.S_ISLNK(item.external_attr >> 16):
                raise SystemExit("Wheel symbolic links are not permitted.")
            if item.is_dir():
                continue
            problem = text_problem(archive.read(item))
            if problem:
                raise SystemExit(f"{item.filename}: {problem}")
    with tarfile.open(sources[0]) as archive:
        seen = set()
        root = None
        for item in archive.getmembers():
            if item.issym() or item.islnk() or not (item.isdir() or item.isfile()):
                raise SystemExit("Source archive contains a link or special file.")
            parts = PurePosixPath(item.name).parts
            if not parts or ".." in parts or PurePosixPath(item.name).is_absolute():
                raise SystemExit("Invalid source archive path.")
            if root is None:
                root = parts[0]
            if parts[0] != root:
                raise SystemExit("Source archive has multiple roots.")
            if not item.isfile():
                continue
            name = '/'.join(parts[1:])
            data = archive.extractfile(item).read()
            problem = text_problem(data) if generated_source(name) else validate(name, data)
            if problem:
                raise SystemExit(f"{name}: {problem}")
            if name == "src/ashrink.egg-info/SOURCES.txt":
                for listed in data.decode().splitlines():
                    if listed not in PUBLIC_FILES and not generated_source(listed):
                        raise SystemExit("Unreviewed source-manifest entry: " + listed)
            seen.add(name)
        required = {p for p in PUBLIC_FILES if not p.startswith('.github/') and p not in {'.gitignore', '.gitattributes'}}
        if not required.issubset(seen):
            raise SystemExit("Source archive omits reviewed source files: " + ', '.join(sorted(required - seen)))
    print("Wheel and source archive satisfy the reviewed public-content manifests.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    paths = set(filter(None, subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")))
    if paths != PUBLIC_FILES:
        raise SystemExit("Tracked source files differ from tools/public-files.txt.")
    for name in sorted(paths):
        file = ROOT / name
        if file.is_symlink():
            raise SystemExit("Source symbolic links are not permitted: " + name)
        problem = validate(name, file.read_bytes())
        if problem:
            raise SystemExit(f"{name}: {problem}")
    # MkDocs publishes every file under docs, including untracked assets.
    for file in (ROOT / "docs").rglob("*"):
        if file.is_symlink():
            raise SystemExit("Documentation symbolic links are not permitted.")
        if file.is_file() and file.relative_to(ROOT).as_posix() not in PUBLIC_FILES:
            raise SystemExit("Unreviewed documentation asset: " + str(file.relative_to(ROOT)))
    print("Tracked source files satisfy the reviewed public-file manifest.")
    if args.archives:
        check_archives(args.archives)


if __name__ == "__main__":
    main()
