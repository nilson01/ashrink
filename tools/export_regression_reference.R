# Optional pinned mr.ash.alpha comparison; not a runtime dependency.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript export_regression_reference.R output.json")
stopifnot(as.character(utils::packageVersion("mr.ash.alpha")) == "0.1.43")
X <- rbind(c(-1,-1),c(-1,1),c(1,-1),c(1,1),
           c(-0.5,0.2),c(0.4,-0.7),c(0.8,0.3),c(-0.3,0.6))
y <- c(-1.8,-0.1,0.6,3.3,0.2,-0.4,1.8,0.7)
make_case <- function(name, update_weights, update_variance, iterations=10000L) {
  fit <- mr.ash.alpha::mr.ash(X,y,sa2=c(0,0.4^2,1.2^2),
      beta.init=c(0,0),pi=c(0.3,0.4,0.3),sigma2=0.7,
      update.pi=update_weights,update.sigma2=update_variance,
      standardize=FALSE,intercept=TRUE,method_q="sigma_indep_q",method_g="caisa",
      max.iter=iterations,tol=list(convtol=1e-12,epstol=1e-15))
  list(name=name, X=unname(X),y=y,estimate_weights=update_weights,
       estimate_residual_variance=update_variance,maxiter=iterations,
       coefficients=as.numeric(fit$beta),intercept=as.numeric(fit$intercept),
       residual_variance=as.numeric(fit$sigma2),weights=as.numeric(fit$pi),iterations=fit$iter)
}
record <- list(reference=list(package="mr.ash.alpha",version=as.character(utils::packageVersion("mr.ash.alpha")),
      commit="8e257fdb73589f280192b1c99a176d82196be298",R=R.version.string,
      namespace=getNamespaceInfo(asNamespace("mr.ash.alpha"),"path")),
      cases=list(make_case("fixed-hyperparameters",FALSE,FALSE),
                 make_case("estimated-weights",TRUE,FALSE),
                 make_case("estimated-variance",FALSE,TRUE),
                 make_case("estimated-both",TRUE,TRUE),
                 make_case("one-sweep",TRUE,TRUE,1L)))
jsonlite::write_json(record,args[1],auto_unbox=TRUE,digits=17,pretty=TRUE)
