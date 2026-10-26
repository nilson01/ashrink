"""Compile the maintained Markdown manual with Pandoc and pdfLaTeX."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/manual/reference.md"
TARGET = SOURCE.with_name("ashrink-reference.pdf")
HEADER = r"""\usepackage{fvextra}
\usepackage{float}
\usepackage{etoolbox}
\BeforeBeginEnvironment{Shaded}{\par\noindent\begin{minipage}{\linewidth}}
\AfterEndEnvironment{Shaded}{\end{minipage}\par\smallskip}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\},fontsize=\small}
\usepackage{microtype}
\usepackage{bookmark}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black,bookmarksdepth=2}
\setlength{\emergencystretch}{3em}
\let\originaltableofcontents\tableofcontents
\renewcommand{\tableofcontents}{{\small\originaltableofcontents}}
\setcounter{tocdepth}{2}
"""
FILTER = r'''function Code(el)
  if FORMAT:match("latex") then
    return pandoc.RawInline("latex", "\\nolinkurl{" .. el.text .. "}")
  end
end
'''


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TARGET)
    target = parser.parse_args().output.resolve()
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        try:
            import pypandoc
            pandoc = pypandoc.get_pandoc_path()
        except (ImportError, OSError) as exc:
            raise SystemExit("Install Pandoc or pypandoc_binary to build the manual.") from exc
    latex = shutil.which("pdflatex")
    if latex is None:
        raise SystemExit("Install a TeX distribution with pdflatex to build the manual.")
    with tempfile.TemporaryDirectory(prefix="ashrink-manual-") as directory:
        work = Path(directory)
        header = work / "header.tex"
        header.write_text(HEADER)
        lua = work / "inline-code.lua"
        lua.write_text(FILTER)
        tex = work / "main.tex"
        subprocess.run([
            pandoc, str(SOURCE), "--from=markdown+tex_math_dollars+fenced_code_attributes",
            "--standalone", "--lua-filter=" + str(lua), "--toc", "--toc-depth=2",
            "--number-sections", "--top-level-division=section", "-V", "documentclass=article",
            "-V", "fontsize=10pt", "-V", "geometry=margin=0.85in", "-V", "colorlinks=true",
            "-V", "linkcolor=black", "-V", "urlcolor=blue", "-V", "toccolor=black", "-V",
            "date=", "--include-in-header=" + str(header),
            "--resource-path=" + str(SOURCE.parent) + ":" + str(ROOT), "-o", str(tex),
        ], check=True)
        content = tex.read_text().replace(
            "../figures/first-example.png", str(ROOT / "docs/figures/first-example.png")
        ).replace(r"\begin{figure}", r"\begin{figure}[H]")
        content = content.replace(r"\section{Overview}", "\\clearpage\n\\section{Overview}")
        tex.write_text(content)
        for _ in range(3):
            result = subprocess.run([latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                                    cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stdout[-6000:])
        shutil.copyfile(work / "main.pdf", target)
    print(target)


if __name__ == "__main__":
    main()
