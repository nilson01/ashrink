"""Fit a headered summary-statistics CSV or export a saved posterior."""

import argparse

from .io import (
    read_csv,
    fit_frame,
    fit_poisson_frame,
    fit_binomial_frame,
    load_result,
    save_result,
    write_csv,
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ashrink")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit scalar summaries or counts from CSV")
    fit.add_argument("input")
    fit.add_argument("output", help="new JSON result file")
    fit.add_argument(
        "--model", choices=["gaussian", "student-t", "poisson", "binomial"], default="gaussian"
    )
    fit.add_argument(
        "--prior-family", choices=["normal", "uniform", "halfuniform"], default="normal"
    )
    fit.add_argument("--estimate", default="estimate")
    fit.add_argument("--se", default="se")
    fit.add_argument("--df", type=float)
    fit.add_argument("--count", default="count")
    fit.add_argument("--exposure", default="exposure")
    fit.add_argument("--successes", default="successes")
    fit.add_argument("--trials", default="trials")
    fit.add_argument("--missing", choices=["raise", "drop"], default="raise")
    export = commands.add_parser("export", help="write posterior summaries to CSV")
    export.add_argument("input")
    export.add_argument("output", help="new CSV summary file")
    export.add_argument("--level", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            write_csv(load_result(args.input), args.output, level=args.level)
            return
        data = read_csv(args.input)
        if args.model in ("gaussian", "student-t"):
            if args.model == "student-t" and args.df is None:
                parser.error("--df is required for Student-t fitting")
            if args.model == "gaussian" and args.df is not None:
                parser.error("use --model student-t with --df")
            result = fit_frame(
                data,
                args.estimate,
                args.se,
                family=args.prior_family,
                df=args.df,
                missing=args.missing,
            )
        elif args.model == "poisson":
            result = fit_poisson_frame(data, args.count, args.exposure)
        else:
            result = fit_binomial_frame(data, args.successes, args.trials)
        save_result(result, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"ashrink: {exc}\n")


if __name__ == "__main__":
    main()
