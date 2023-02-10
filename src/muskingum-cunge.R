#http: /  / ponce.sdsu.edu / muskingum_cunge_method_with_variable_parameters.html

#The Muskingum - Cunge method with variable parameters is applied herein to the
#problem posed by Thomas (8) in his classical paper on flood routing. The problem
#consists of tracking the travel and subsidence of a flood wave of sinusoidal shape
#in a unit width channel with a steady -
#state rating curve given by:

#rating curve
d=seq(0,100)
q=0.668*d^(5/3)

#inflow hydrograph
time1=seq(0,96)
time2=seq(97,200)
inflow1 = 125 - 75*cos(pi*time1/48)
inflow2=rep(x=50,length(time2))
inflow=c(inflow1,inflow2)
time=c(time1,time2)
plot(time,inflow)

DX=25 #miles
DT=6 #hours
celerity = 1 #m/s?
K=DX/celerity
getX=function(q,S,celerity,DX){
  return (
    0.5*(1-(q/(S*celerity*DX)))
    )
  
}
courant=  celerity*DT/DX
reynolds= (qref/S)/(celerity*DX)

C1 = ((DT/K)+2*X)/(2*(1-X)+(DT/K))
C2=((DT/K)-2*X)/(2*(1-X)+(DT/k))
C3=(2*(1-X)-DT/K)/(2*(1-X)+DT/K)
