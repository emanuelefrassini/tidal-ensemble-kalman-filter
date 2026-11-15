function [x_est,obs_values_1,obs_values_2,obs_values_3,obs_values_4,x_estbar ,yobs,H,Nk,K,Ex,Px,x_ini,x_forecast,y_for]=ensemblekfilter_2(settings,num_members,tfin) 
%[x_tr,x_estbar,ybar]= ensemblekfilter(settings) 
loc_names=settings.loc_names;
nseries=4;%5;%length(loc_names);
k=settings.n;
ilocs=settings.ilocs;
p1=200;
t=settings.t;

% x_est=zeros(p1,num_members,length(t));
x_est=zeros(p1,num_members,length(t));

%get H matrix
 H=sparse(nseries,2*k);
%H=sparse(nseries,2*k+1);

 j=1;
 for i=1:4%5%length(ilocs)
          H(j,ilocs(i+1))=1;
    j=j+1;
 end
w=0.001;
z=0.001; 
h=0.001;
       rng(3);
P=0.1*eye(p1);%0.1
S=chol(P);
x_ini=zeros(2*k,num_members);
     for i=1:num_members
% 	x_ini(1:2:end,i)=settings.h_0;
%     x_ini(2:2:end,i)=settings.u_0;
%  x_ini(1:2:end,i)=w*randn;
%  x_ini(2:2:end,i)=0;
%x_ini(:,i)=w*randn(p1,1);
x_ini(:,i)=S'*rand(p1,1);
     end

	 
  

m1=size(H,1);

Zcov=eye(m1); %create measurement noise covariance matrix
for j=1:m1
  Zcov(j,j)=0.0000001;%0.0000001;%z^(2.3);%z^2;
end
%[x,A,B]=wave1d_timestep(x_ini(:,1),i,settings);

A=settings.A;
B=settings.B;
%        M=A\B;
%        c=zeros(size(M,1),1);
%        c(1)=1;
%        M=[M c];
%         r=zeros(size(M,2),1);
%         r(end)=0.9726;%alpha;
      
        Nk=0;%.303;%0.2853;%0.2853; 
%         M=[M;r']; %timestep matrix with addition of stochastic
%   add=zeros(size(M,2),1);

%        [~,obs_values_1]=wave1d_read_series('tide_vlissingen.txt');
%     [~,obs_values_2]=wave1d_read_series('tide_terneuzen.txt');
%    [~,obs_values_3]=wave1d_read_series('tide_hansweert.txt');
%    [~,obs_values_4]=wave1d_read_series('tide_bath.txt');
 [~,obs_values_1]=wave1d_read_series('waterlevel_vlissingen.txt');
       [~,obs_values_2]=wave1d_read_series('waterlevel_terneuzen.txt');
    [~,obs_values_3]=wave1d_read_series('waterlevel_hansweert.txt');
   [~,obs_values_4]=wave1d_read_series('waterlevel_bath.txt');
  % [~,obs_values_5]=wave1d_read_series('.txt');
  j=1;
%        x_ini(ilocs(2),:)=obs_values_1(1);
%        x_ini(ilocs(3),:)=obs_values_2(1);
%        x_ini(ilocs(4),:)=obs_values_3(1);
%        x_ini(ilocs(5),:)=obs_values_4(1);

   for i=1:tfin%length(t) 
               wk=w*randn;
       % add(end)=wk;
        Px=zeros(p1,p1);
        x_forecast=zeros(p1,num_members);
        A=settings.A;
B=settings.B;
        for ind=1:num_members
             rhs=B*x_ini(:,ind);
             rhs(1)=settings.h_left(i);
x_forecast(:,ind)=A\rhs+Nk;
        %x_forecast(:,ind)=M*[x_ini(:,ind);Nk]+add; %the last element of x_forecast(:,ind) contains Nk+1
        end
        % x_estbar=mean(x_forecast(1:end-1,:),2); 
      x_estbar=mean(x_forecast,2); 
      
         for ii=1:num_members
Ex=x_forecast(:,ii)-x_estbar;
Px=Px+Ex*Ex';
         end
         Px=Px/(num_members-1);
         
             Kk=H*Px*H'+Zcov;
             Kkk=(Px*H');
             K=Kkk*inv(Kk);%Kkk/Kk;
        obs=zeros(nseries,1);
        obs(1)=obs_values_1(i);
        obs(2)=obs_values_2(i);
        obs(3)=obs_values_3(i);
        obs(4)=obs_values_4(i);
       % obs(5)=obs_values_5(i);
y_for=zeros(nseries,num_members);
y=zeros(nseries,num_members);
Z=zeros(m1,num_members);

       for index=1:num_members
           Z(:,index)=h.*randn(m1,1);
          y(:,index)=obs+Z(:,index);
        %  y_for(:,index)=H*x_forecast(1:p1,index);
            y_for(:,index)=H*x_forecast(:,index); 

       end
    x_est(:,:,i)=x_forecast+K*(y-y_for);
   
%x_est(:,:,i)=x_forecast+K*(y-y_for);
   yobs(:,:,i)=y(:,:);
   Nk=0.9726*Nk+wk;
   x_ini=x_est(:,:,i);
%   if i==1
%    plot(K(:,1))
%    hold on
%    plot(K(:,2))
%    plot(K(:,3))
%    plot(K(:,4));
%   end
   end
%   for ind=1:num_members
%    x_ini(:,ind)=mean(x_est(:,:,i),2);
%   end
   %x_ini=zeros(2*k,num_members);
% for i=1:num_members
% x_ini(:,i)=S'*rand(p1,1);
% end
% %    end
   for i=tfin+1:length(t)
        %x_forecast=zeros(p1,num_members);
        for ind=1:num_members
             rhs=B*x_ini(:,ind);
             rhs(1)=settings.h_left(i);
x_forecast(:,ind)=A\rhs+Nk;
        end
  
x_est(:,:,i)=x_forecast;
   x_ini=x_est(:,:,i);
   wk=w*randn;
   Nk=0.9726*Nk+wk;
   end



