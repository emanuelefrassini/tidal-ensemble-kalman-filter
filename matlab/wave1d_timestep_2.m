function newx=wave1d_timestep_2(x,i,settings,n)
%function newx=timestep(x,i,settings)
%return (h,u) one timestep later
%take one timestep
    A=settings.A;
    B=settings.B;
    rhs=B*x; %B*x
    rhs(1)=settings.h_left(i)+n; %left boundary, take at t+dt HERE I PUT THE STOCHASTIC FORCING TERM
    newx=A\rhs;
    %N(k+1)=alpha*N(k)+w(k)
%w(k) = normrnd( mu , sigma ) 
