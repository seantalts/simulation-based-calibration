generate_data = function(spde, A.gen, N,m )
{
  #generate field from pcmatern
  #NB: tHis REALLY assumes spde was made with inla.spde2.pcmatern
  #It will fail WEIRDLY if this is not true!!!!
  params = spde$f$hyper.default$theta1$param[1:2]
  lambda_range = params[1]
  lambda_sigma = params[2]
  
  sigma = rexp(1,lambda_sigma)
  range = 1/rgamma(1,shape=1,rate=lambda_range)
#browser()
  Q = INLA::inla.spde2.precision(spde,c(log(range),log(sigma)))
  
  
  field = as.numeric(INLA::inla.qsample(1,Q))
  
  #generate logit intercept
  #quantiles of inv.logit(rnorm(100000,-2.5,1.5))
  #This is weakly informative based on global incence from 0.3%-20%
  #incidence in Kenya is 5-7% (depending on the year (5% in 2016, 7% in 2003))
  #         1%         10%         50%         90%         99% 
  #  0.002469285 0.011752410 0.075601298 0.357199965 0.727354083 
  logit_intercept = rnorm(1,-2.5,1.5)
  
  
  ## generate iid noise
  # PC prior with alpha=0.05, U=1
  ##generate data
  lam_iid = -log(0.05)/1
  sigma_iid = rexp(1,lam_iid)
  v = rnorm(sum(m), sd = sigma_iid)
  
  
  ## generate logit_p
  
  logit.prev= rep( rep(logit_intercept, 
                       length(dim(loc.data)[1]))+as.numeric(A.gen%*%field),times=m) + v
  
  y = rbinom(sum(m),N,exp(logit.prev)/(1+exp(logit.prev)))
  
  
  
  return(
    list(
      hyper = list(log_range=log(range),log_sigma = log(sigma),log_prec = -2*log(sigma_iid)),
      logit_intercept = logit_intercept,
      field = field,
      v=v,
      y=y
         )
  )
}

functionals = function(field,mesh,points.mc) {
  
  ## Field may be a matrix!
  
  K  =length(points.mc)
  out = list()
  
  inv.logit = function(x) {exp(x)/(1+exp(x))}
  
  
  for ( k in 1:K ) {
    A.mc = inla.spde.make.A(mesh,points.mc[[k]])
    int_vals = inv.logit(A.mc%*%field)
    out[[k]] = apply(X=int_vals,MARGIN=2,FUN = mean)
  }
  
  return(out)
}

rank_functionals = function(y,N,spde,A.est,m,functional_true,inla_seed) {
  
  ## Set up INLA formula and data
  formula = y ~ -1 + intercept + f(field, model=spde) + f(eps,model="iid",hyper = list(prec=list(hyper="pc.prec",param=c(1,0.05)))) 
  spde.index = inla.spde.make.index("field",spde$n.spde)
  stack.est = inla.stack( data = list(y = y,N=N), A = list(A.est,1), effects=list(c(spde.index,list(intercept=1)), list(eps = seq_len(length(y)) )))
  
  
  #This takes about 1 minute
  result = inla(formula, family="binomial", Ntrials=N, 
                data=inla.stack.data(stack.est), 
                control.predictor = list(compute=TRUE,
                                         A=inla.stack.A(stack.est)),
                control.fixed = list(mean.intercept = -2.15 ,prec.intercept =1.5^(-2) ),
                verbose=FALSE,control.compute = list(config=TRUE),num.threads = 1)
  
  
  #This takes about 10 secs
  samps = inla.posterior.sample(100,result,seed = inla_seed)
  index.field=grep(x = row.names(samps[[1]]$latent),pattern="field*")
  index.intercept = which(row.names(samps[[1]]$latent)=="intercept")
  
  ## I hope this works
  
  logit_prev = sapply(X=samps, FUN = function(samp) as.numeric(samp$latent[index.field]+ samp$latent[index.intercept]))
  functionals = (functionals(logit_prev,mesh,points.mc))
  
  rank = rep(NA,length(functional_true))
  for(i in 1:length(functional_true)) {
    rank[i] = sum(functionals[[i]] < functional_true[[i]])
  }
  
  return(rank)
}


sbc = function() {
  dat = generate_data(spde, A.gen, N, m)
  logit_prev_true = dat$logit_intercept+ dat$field
  functional_true = functionals(logit_prev_true,mesh,points.mc)
  ranks = rank_functionals(dat$y,N,spde,A.est,m,functional_true,0L)
  return(ranks)
}

