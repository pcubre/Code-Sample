###Horseshoe Probit Regression
n<-10000
beta <- c(1.3,4,-1,1.6,5.-2,0,0)
p <- length(beta)
X<-rnorm(n*p)
X<-matrix(X,ncol = p)
mu <- X %*% beta
Y <- rbinom(n,1,pnorm(mu))

gibbs <- function(X,Y){
  n <- dim(X)[1]
  p <- dim(X)[2]
  .C("gibbs_sampler",
     n = as.integer(n),
     p = as.integer(p),
     X = as.double(X),
     Y = as.integer(Y),
     store = as.double(rep(0,p*100)))$store
}


