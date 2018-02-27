/**********************************************
Copyright (c) 2018 pcubre. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must 
display the following acknowledgement: 
This product includes software developed by pcubre.
4. Neither the name of pcubre nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY PCUBRE "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PCUBRE
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************/



sample_tau <- function(p) {
  local_array <- rep(0,32);
  d_tau_X <- rep(0,32);
  flag <- 1;
  d_tau_square_inverse <- 0;
  d_shape <-(p+1)/2;
  d_scale <- e_inverse + (sum(beta^2*lambda_square_inverse)) / 2;
  d_u_1 <- 0;
  d_u_2 <- 0;
  d_V <- 0;
  a = (2 * d_shape - 1)^(-.5);
  b = d_shape - log(4.0);
  c = d_shape + 1 / a;
  

  while (flag) {
    for(idx in 32:1){
      d_u_1 = runif(1)
      d_u_2 = runif(1)
      d_V = a * log(d_u_1 / (1 - d_u_1));
      d_tau_X[idx] = d_shape * exp(d_V);
      if (b + c*d_V - d_tau_X[idx] >= log(d_u_1*d_u_1*d_u_2)) {
        local_array[idx] = 0;
      } else {
        local_array[idx] = 1;
      }
    }
    if (idx == 1) {
      for (i in 1:32) {
        #cat(i)
        #cat("\n")
        if (local_array[i] == 1) {
          d_tau_square_inverse = (d_scale / d_tau_X[i]);
          flag = 0;
          break;
        }
      }
    }
  }
  return(d_tau_square_inverse);
}


sample_tau_square_inverse <- function(p){
  u <- runif(2);
  shape <- (p+1)/2;
  #note for gamma distribution this is the rate
  inverse_gamma_scale_gamma_rate <- e_inverse + (sum(beta^2*lambda_square_inverse)) / 2;
  a <- (2 * shape -1)^(-.5)
  b <- shape - log(4)
  c <- shape + 1/a;
  V <- a * log(u[1]/(1-u[1]));
  X <- shape * exp(V);
  while( b + c * V - X >= log(u[1]^2*u[2])){
    u <- runif(2);
    V <- a * log(u[1]/(1-u[1]));
    X <- shape * exp(V);
  }
  return(inverse_gamma_scale_gamma_rate * X^(-1));
}

n<-10^4
v_sample_tau <- Vectorize(sample_tau)
v_sample_tau_square_inverse <- Vectorize(sample_tau_square_inverse)
t_shape <- (p+1)/2;
t_scale <- (e_inverse + (sum(beta^2*lambda_square_inverse)) / 2);
t<-1/rgamma(n,shape = t_shape, rate = t_scale);
t<-v_sample_tau(rep(p,n))
h<-hist(t,nclass = 20)
plot(h,col = "grey")
xlines <- seq(min(h$breaks),max(h$breaks),length.out = n)
lines(x = xlines, y = diff(h$breaks)[1]*n*t_scale^t_shape*xlines^(-t_shape-1)*exp(-t_scale/xlines)/gamma(t_shape))
#plot(t_scale^t_shape/gamma(t_shape)*xlines^(-t_shape-1)*exp(-t_scale/xlines))


sample_z <- function(Y,mu){
  y_star <- 2 * Y - 1;
  mu_star <- mu * y_star;
  a <- -mu_star;
  if(a < 0){
    cat(1)
    z <- rnorm(1,0,1);
    while(z < a){
      z <-rnorm(1,0,1);
    }
  } else if (a < 0.257) {
    cat(2)
    z <- rnorm(1,0,1);
    while(z < a && -z < a){
      z <- rnorm(1,0,1);
    }
    if(z < 0){ z <- -z;}
  } else {
    cat(3)
    u_1 <- runif(1);
    u_2 <-runif(1);
    lambda_star <- (a + sqrt(a^2+4))/2;
    z <- -log(u_1)/lambda_star + a;
    while( u_2 > exp(-(z-lambda_star)^2/2)){
      u_1 <- runif(1);
      u_2 <-runif(1);
      z <- -log(u_1)/lambda_star + a;
    }
  }
  z <- (z+mu_star) * y_star;
  return(z);
}
v_sample_z <- Vectorize(sample_z)




n<-10^5
mu <- 0
z<-v_sample_z(rep(1,n),mu)
h<-hist(z)
plot(h,col = "grey")
xlines <- seq(min(h$breaks),max(h$breaks),length.out = n)
lines(x = xlines, y = dnorm(xlines,mu,1)*n*diff(h$breaks)[1]/(1-pnorm(0-mu)))
lines(x = xlines, y = dnorm(xlines,mu,1)*n*diff(h$breaks)[1]/(pnorm(0-mu)))




##################
# Simulation setup
#################
n <- 10000;
beta <- c(1.3,4,-1,1.6,5,-2,0,0,0,0);
p <-length(beta);
X<-matrix(rnorm(p*n),ncol = p);
mu <- X%*%beta; 
Y <- rbinom(n,1,pnorm(mu));
# table(sign(mu),Y);

#################
# Sampler Init
#################
tau_square_inverse = 1.0;
Lambda_square_inverse = diag(1,p);
lambad_square_inverse = rep(1,p);
v_inverse = rep(1,p);
e_inverse = 1.0;
z <- rep(0.0,n);
iter <- 10^4
Beta <- matrix(rep(NA,p * iter),ncol = p)
for(t in 1:iter){
  # Sample beta
  XTX <- t(X) %*% X;
  Sigma_inverse <- XTX + tau_square_inverse * Lambda_square_inverse;
  R_inverse <- chol(Sigma_inverse);
  s <- rnorm(p);
  v <- solve(R_inverse, s);
  beta <- t(X) %*% z;
  beta <- solve(Sigma_inverse, beta);
  beta <- v + beta;
  # Sample z
  mu <- X %*% beta;
  z<-v_sample_z(Y,mu);
  # sample lambda^(-2)
  u <- runif(p);
  lambda_square_inverse <- -log(u)/(v_inverse+tau_square_inverse*beta^2/2);
  Lambda_square_inverse <- diag(lambda_square_inverse);
  # sample v^(-2)
  u <- runif(p);
  v_inverse <- -log(u)/(1+lambda_square_inverse);
  # sample e^(-1)
  u <- runif(1);
  e_inverse <- -log(u)/(1+tau_square_inverse);
  #sample tau^(-2)
  t_shape <- (p+1)/2;
  t_scale <- (e_inverse + (sum(beta^2*lambda_square_inverse)) / 2);
  tau_square_inverse <- sample_tau_square_inverse(p)
  #store
  #if(t %% 10 == 0){
    Beta[t,] <- beta
  #}
}
rbind(apply(Beta[iter/2:iter,],2,mean)-c(1.3,4,-1,1.6,5,-2),apply(Beta2[iter/4:iter,],2,mean)-c(1.3,4,-1,1.6,5,-2),c(1.3,4,-1,1.6,5,-2))

