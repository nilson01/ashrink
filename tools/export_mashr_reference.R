# Optional installed multivariate reference; not required by AShrink at runtime.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop("Usage: Rscript export_mashr_reference.R output.json")
stopifnot(as.character(utils::packageVersion("mashr"))=="0.2.79")
B <- rbind(c(1,-.5),c(.2,1.3),c(-.8,-.4),c(.1,.2),c(1.4,1.1),c(-1.2,.8))
S <- rbind(c(.5,.7),c(.6,.4),c(.8,.5),c(.4,.6),c(.7,.7),c(.6,.8))
V <- matrix(c(1,.2,.2,1),2)
make_case <- function(name,U,pointmass=TRUE,weights=NULL) {
 data <- mashr::mash_set_data(B,S,alpha=0,V=V)
 if(is.null(weights)) {
  fit <- mashr::mash(data,Ulist=U,grid=1,normalizeU=FALSE,usepointmass=pointmass,
                     prior="uniform",verbose=FALSE,outputlevel=3,output_lfdr=TRUE,pi_thresh=0)
 } else {
  g<-list(pi=weights,Ulist=U,grid=1,usepointmass=pointmass)
  fit <- mashr::mash(data,g=g,fixg=TRUE,verbose=FALSE,outputlevel=3,output_lfdr=TRUE,pi_thresh=0)
 }
 list(name=name,effects=unname(B),standard_errors=unname(S),correlation=unname(V),
      patterns=lapply(U,unname),point_mass=pointmass,fixed=!is.null(weights),
      weights=as.list(as.numeric(fit$fitted_g$pi)),posterior_mean=unname(ashr::get_pm(fit)),
      posterior_sd=unname(ashr::get_psd(fit)),lfsr=unname(ashr::get_lfsr(fit)),
      lfdr=unname(ashr::get_lfdr(fit)),
      posterior_covariance=lapply(seq_len(nrow(B)),function(i) unname(fit$result$PosteriorCov[,,i])))
}
record<-list(reference=list(package="mashr",version=as.character(utils::packageVersion("mashr")),
 R=R.version.string,namespace=getNamespaceInfo(asNamespace("mashr"),"path")),
 cases=list(make_case("full-rank",list(matrix(c(1.2,.4,.4,.8),2)),FALSE),
            make_case("estimated-mixture",list(diag(2),matrix(1,2,2))),
            make_case("fixed-singular-mixture",list(matrix(1,2,2),diag(c(1,0))),TRUE,c(.2,.4,.4))))
jsonlite::write_json(record,args[1],auto_unbox=TRUE,digits=17,pretty=TRUE)
