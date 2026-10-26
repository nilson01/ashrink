# Optional installed-reference check. R is not required by AShrink.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript export_ashr_reference.R output.json")
if (!requireNamespace("ashr", quietly = TRUE) ||
    !requireNamespace("jsonlite", quietly = TRUE)) stop("Install ashr 2.2-63 and jsonlite first")
stopifnot(as.character(utils::packageVersion("ashr")) == "2.2.63")
y <- c(-1.4, -0.6, -0.1, 0, 0.2, 0.7, 1.8, 2.2)
se <- c(0.7, 0.4, 0.5, 0.3, 0.5, 0.4, 0.8, 0.7)
normal <- ashr::normalmix(c(0.3, 0.4, 0.3), rep(0, 3), c(0, 0.4, 1.2))
uniform <- ashr::unimix(c(0.3, 0.4, 0.3), c(0, -0.6, -1.8), c(0, 0.6, 1.8))
half <- ashr::unimix(c(0.2, 0.15, 0.25, 0.15, 0.25),
                    c(0, -0.6, -1.8, 0, 0), c(0, 0, 0, 0.6, 1.8))
case <- function(name, family, g, df = NULL, fixed = TRUE) {
  a <- ashr::ash(y, se, g = g, fixg = fixed, mixcompdist = family,
                 prior = "uniform", df = df, control = list(verbose = FALSE))
  list(name = name, family = family, df = df, fixed = fixed, effects = y,
       standard_errors = se, prior = unclass(g),
       posterior_mean = ashr::get_pm(a), posterior_sd = ashr::get_psd(a),
       lfdr = ashr::get_lfdr(a), lfsr = ashr::get_lfsr(a),
       weights = ashr::get_fitted_g(a)$pi)
}
cases <- list(case("normal-fixed", "normal", normal),
              case("uniform-fixed", "uniform", uniform),
              case("halfuniform-fixed", "halfuniform", half),
              case("t-uniform-fixed", "uniform", uniform, df = 5),
              case("t-halfuniform-fixed", "halfuniform", half, df = 5),
              case("normal-estimated", "normal", normal, fixed = FALSE))
record <- list(reference = list(package = "ashr", version = as.character(utils::packageVersion("ashr")),
                               R = R.version.string, namespace = getNamespaceInfo(asNamespace("ashr"), "path")),
               cases = cases)
jsonlite::write_json(record, args[1], auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null")
