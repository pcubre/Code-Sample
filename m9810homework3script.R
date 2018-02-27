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


#PRoblem 1
getwd()
setwd('m9810Fall2014')
school1=scan("school1.dat")
school2=scan("school2.dat")
school3=scan("school3.dat")
#use books definitions
k_0=1
mu_0=5
s2_0=4
nu_0=2
n.1=length(school1)
n.2=length(school2)
n.3=length(school3)
#posterior variables
k_n.1=k_0+n.1
k_n.2=k_0+n.2
k_n.3=k_0+n.3
nu_n.1=nu_0+n.1
nu_n.2=nu_0+n.2
nu_n.3=nu_0+n.3
ybar.1=mean(school1)
s2.1=var(school1)
ybar.2=mean(school2)
s2.2=var(school2)
ybar.3=mean(school3)
s2.3=var(school3)
s2_n.1 = (nu_0*s2_0 + (n.1-1)*s2.1 + k_0*n.1*(ybar.1-mu_0)^2/(k_n.1))/nu_n.1
mu_n.1=(n.1*ybar.1+k_0*mu_0)/k_n.1
s2_n.2 = (nu_0*s2_0 + (n.2-1)*s2.2 + k_0*n.1*(ybar.2-mu_0)^2/(k_n.2))/nu_n.2
mu_n.2=(n.2*ybar.2+k_0*mu_0)/k_n.2
s2_n.3 = (nu_0*s2_0 + (n.3-1)*s2.3 + k_0*n.3*(ybar.3-mu_0)^2/(k_n.3))/nu_n.3
mu_n.3=(n.3*ybar.3+k_0*mu_0)/k_n.3
#monte carlo
iter=1e6
postsigma2.1 = 1/rgamma(iter,nu_n.1/2,rate=s2_n.1*nu_n.1/2)
posttheta.1 = rnorm(iter,mu_n.1,(postsigma2.1/k_n.1)^.5)
newy.1=rnorm(iter,posttheta.1,(postsigma2.1/k_n.1)^.5)
postsigma2.2 = 1/rgamma(iter,nu_n.2/2,rate=s2_n.2*nu_n.2/2)
posttheta.2 = rnorm(iter,mu_n.2,(postsigma2.2/k_n.2)^.5)
newy.2=rnorm(iter,posttheta.2,(postsigma2.2/k_n.2)^.5)
postsigma2.3 = 1/rgamma(iter,nu_n.3/2,rate=s2_n.3*nu_n.3/2)
posttheta.3 = rnorm(iter,mu_n.3,(postsigma2.3/k_n.3)^.5)
newy.3=rnorm(iter,posttheta.3,(postsigma2.3/k_n.3)^.5)

#mean of posterior theta and 95%
mean(posttheta.1)
mean(posttheta.2)
mean(posttheta.3)
quantile(posttheta.1,c(.025,.0975))
quantile(posttheta.2,c(.025,.0975))
quantile(posttheta.3,c(.025,.0975))

#mean of sigma
mean((postsigma2.1)^.5)
mean((postsigma2.2)^.5)
mean((postsigma2.3)^.5)
quantile((postsigma2.1)^.5,c(.025,.0975))
quantile((postsigma2.2)^.5,c(.025,.0975))
quantile((postsigma2.3)^.5,c(.025,.0975))
          
#P(thi<thj<thk)
pth=rep(NA,6)
#123
pth[1]=sum(posttheta.1<posttheta.2 & posttheta.2 < posttheta.3)/iter
#132
pth[2]=sum(posttheta.1<posttheta.3 & posttheta.3 < posttheta.2)/iter
#213
pth[3]=sum(posttheta.2<posttheta.1 & posttheta.1 < posttheta.3)/iter
#231
pth[4]=sum(posttheta.2<posttheta.3 & posttheta.3 < posttheta.1)/iter
#312
pth[5]=sum(posttheta.3<posttheta.1 & posttheta.1 < posttheta.2)/iter
#321
pth[6]=sum(posttheta.3<posttheta.2 & posttheta.2 < posttheta.1)/iter
pth

#P(yi<yj<yk)
pnewy=rep(NA,6)
#123
pnewy[1]=sum(newy.1<newy.2 & newy.2 < newy.3)/iter
#132
pnewy[2]=sum(newy.1<newy.3 & newy.3 < newy.2)/iter
#213
pnewy[3]=sum(newy.2<newy.1 & newy.1 < newy.3)/iter
#231
pnewy[4]=sum(newy.2<newy.3 & newy.3 < newy.1)/iter
#312
pnewy[5]=sum(newy.3 <newy.1 & newy.1 < newy.2)/iter
#321
pnewy[6]=sum(newy.3 <newy.2 & newy.2 < newy.1)/iter
pnewy

#y1 biggest
pnewy[6]+pnewy[4]
#th1 biggest
pth[6]+pth[4]

###Problem no 2
setA=scan('menchild30bach.dat')
setB=scan('menchild30nobach.dat')
yA=sum(setA)
yB=sum(setB)
nA=length(setA)
nB=length(setB)
ath=2
bth=1
abg=c(8,16,32,64,128)

## Use Gibbs sampler to sample theta and gamma
iter = 5e4; ## number of iterations (i.e., T)

## Save records
Th = rep(NA, iter);
Ga = rep(NA, iter);
ThM= rep(NA,5);
GaM= rep(NA,5);
FinalM=rep(NA,5);
for(i in 1:5){
#initiate sampler
th=rgamma(1,2,rate=1)
ga=rgamma(1,abg[i],rate=abg[i])

Th[1]=th
Ga[1]=ga
## Start the Gibbs sampler
for(t in 2:iter){
  #use previous generations ga
  th = rgamma(1,yA+yB+ath ,rate=bth+nA+nB*ga);
  #use this generations th
  ga = rgamma(1, yB+abg[i] ,rate=abg[i]+nB*th);
  
  Th[t] = th;
  Ga[t] = ga;
}

#plot(Th, typ = 'l')
#plot(Ga, typ = 'l')

ThM[i]=mean(Th[-(1:500)])
GaM[i]=mean(Ga[-(1:500)])
FinalM[i]=sum(Th[-(1:500)]*(Ga[-(1:500)]-1))/(iter-500)
}
FinalM
plot(1:5,FinalM)

###Problem no 3
glu=scan('glucose.dat')
hist(glu, prob = TRUE,xlab="Glucose Concentration", main = "Glucose Concentrations in Women", breaks = 25);
dglu<-density(glu)
lines(dglu)
#initial variables
a=1
b=1
mu_0=120
t2_0=200
s2_0=1000
nu_0=10
n=length(glu)
ybar=mean(glu)
s2=var(glu)
#number of samples
iter=1e5
#storage
Th.1=rep(NA,iter)
Th.2=rep(NA,iter)
S2.1=rep(NA,iter)
S2.2=rep(NA,iter)
P=rep(NA,iter)
X=matrix(,nrow=n,ncol=iter)
newy=rep(NA,iter)
#initiate sampler
p=rbeta(1,a,b)
P[1]=p
newx=rbinom(1,1,p)
x=rbinom(n,1,p)
#x=sample(c(0,1),replace=TRUE,n,prob=c(p[1],1-p[1]))
X[,1]=x
s2.1=1/rgamma(1,nu_0/2,rate=nu_0*s2_0/2)
S2.1[1]=s2.1
s2.2=1/rgamma(1,nu_0/2,rate=nu_0*s2_0/2)
S2.2[1]=s2.2
th.1=rnorm(1,mu_0,t2_0)
th.2=rnorm(1,mu_0,t2_0)
Th.1[1]=min(th.1,th.2)
Th.2[1]=max(th.1,th.2)
if (newx==1){
  newy[1]=rnorm(1,th.1,sd=sqrt(s2.1))
}else
{
  newy[1]=rnorm(1,th.2,sd=sqrt(s2.2))
}
for(i in 2:3){
  p<-rbeta(1,a+n-sum(x),b+sum(x))
  for(j in 1:n){
    pp1<-p*exp(-((glu[j]-th.1)^2)/(2*s2.1))/sqrt(s2.1);
    pp2<-(1-p)*exp(-((glu[j]-th.2)^2)/(2*s2.2))/sqrt(s2.2);
    ptemp=pp1/(pp1+pp2); 
    print(ptemp)
    x[j]=rbinom(1,1,ptemp)
  }
  s2.1=1/rgamma(1,(n-sum(x)+nu_0)/2,rate=(sum((1-x)*(glu-th.1)^2)+nu_0*s2_0)/2)
  s2.2=1/rgamma(1,(sum(x)+nu_0)/2,rate=(sum(x*(glu-th.2)^2)+nu_0*s2_0)/2)
  th.1=rnorm(1,(sum(glu*(1-x))+mu_0)/(sum(1-x)*t2_0+s2.1),sd=sqrt(s2.1*t2_0/(sum(1-x)*t2_0+s2.1)))
  th.2=rnorm(1,(sum(glu*x)+mu_0)/(sum(x)*t2_0+s2.2),sd=sqrt(s2.2*t2_0/(sum(x)*t2_0+s2.2)))

  
  P[i]=p
  X[,i]=x
  S2.1[i]=s2.1
  S2.2[i]=s2.2
  Th.1[i]=min(th.1,th.2)
  Th.2[i]=max(th.1,th.2)
}
for (i in 2:iter){
  newx<-rbinom(1,1,P[i])
  if (newx==0){
    newy[i]=rnorm(1,Th.1[i],sd=sqrt(S2.1[i]))
  }else
  {
    newy[i]=rnorm(1,Th.2[i],sd=sqrt(S2.2[i]))
  }
}

library('coda')
Th11=as.mcmc(Th.1[-(1:1000)])
traceplot(Th11)
acf(Th11)
Th22=as.mcmc(Th.2[-(1:1000)])
traceplot(Th22)
acf(Th22)
hist(newy, prob = TRUE,xlab="Glucose Concentration", main = "Glucose Concentrations in Women posterior density", breaks = 25);
dnewy<-density(newy)
lines(dnewy)
NEWYY=as.mcmc(newy)
traceplot(NEWYY)
x
