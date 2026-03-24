close all;
clc;
clear all;

ti=0;
tf=10;
tspan=[ti tf];
x0=[1.2,1.2,1.2]';

global Wo;
global A1; 
global A2;
A1=3;
A2=20;



Wo =   [31.33055027816361
   2.93884897165486
   4.75102477974928
  19.56980768435391
  15.40755550561783
   4.88659432012399
  -1.01969903632577
   0.00762466039624
   0.69402940708167
  -0.02266376799493
   5.78421754968307
   0.57325092978855
   4.53208457019513
   1.41686140809859
   3.73371666772685
  -0.41516497890503
   4.90036450112178
   0.08246764008647
   3.18881180573869
   1.04263826845779
   0.15205757661334];


[t,x]= ode23('filelin3x3',tspan,x0,odeset('MaxStep',1e-1));

% Plotting the results
figure(1);
hold on;

plot(t,x(:,1),'r-');plot(t,x(:,2),'b:');plot(t,x(:,3),'g-.');
xlabel('Time(s)');ylabel('Systems States');legend('x1','x2','x3');
%title('State Trajectory for LQR Control Law');
%title('State Trajectory for LQR Control Law with Bounds');
title('State Trajectory for the Nearly Optimal Control Law');

for i=1:length(x)
    
    x1=x(i,1);    x2=x(i,2);    x3=x(i,3);
	dPHI=[ 	            2*x1            0               0
       	        		0               2*x2            0
          	     		0               0               2*x3
             	  		x2              x1              0
               			x3              0               x1
		           		0               x3              x2  
       	        		4*x1^3          0               0
          	     		0               4*x2^3          0
             	  		0               0               4*x3^3
               			2*x1*x2^2       2*x1^2*x2       0
		           		2*x1*x3^2       0               2*x1^2*x3
       	        		0               2*x2*x3^2       2*x2^2*x3
          	     		2*x1*x2*x3      x1^2*x3         x1^2*x2
             	  		x2^2*x3         2*x1*x2*x3      x1*x2^2   
               			x2*x3^2         x1*x3^2         2*x1*x2*x3
		           		3*x1^2*x2       x1^3            0
       	        		3*x1^2*x3       0               x1^3
          	     		x2^3            3*x1*x2^2       0
             	  		x3^3            0               3*x1*x3^2
		           		0               x3^3            3*x2*x3^2
                        0               3*x2^2*x3       x2^3		];   
                    
	g = [ 0  0;
        0  1;
        1 0];

   R=eye(2);
   
   uu = -0.5*inv(R)*g'*dPHI'*Wo;
   u = [A1*tanh(1/A1*uu(1));
        A2*tanh(1/A2*uu(2))];    
   U(i,:)=[u(1),u(2)];
   
   %U(i,:) = [(-8.3056*x1-2.2827*x2-4.6607*x3)   (-8.5707*x1-2.7323*x2-2.2827*x3)];   
 
     if U(i,1)>=A1
         U(i,1)=A1;
     elseif U(i,1)<=-A1
         U(i,1)=-A1;
     end
     if U(i,2)>=A2
         U(i,2)=A2;
     elseif U(i,2)<=-A2
         U(i,2)=-A2;
     end
end
%  clear U
%   U = [A1*tanh((1/A1)*(-8.3056*x(:,1)-2.2827*x(:,2)-4.6607*x(:,3))),...
%        A2*tanh((1/A2)*(-8.5707*x(:,1)-2.7323*x(:,2)-2.2827*x(:,3)))];
%  U = [(-8.3056*x(:,1)-2.2827*x(:,2)-4.6607*x(:,3)),...
%          (-8.5707*x(:,1)-2.7323*x(:,2)-2.2827*x(:,3))];                


figure(2);
hold on;
plot(t,U(:,1),'r-');plot(t,U(:,2),'b:');
xlabel('Time(s)');ylabel('Control Input u(x)');legend('u1','u2');
%title('LQR Control Signal');
%title('The Initial Stabilizing Control: LQR Control Signal with Bounds');
title('Nearly Optimal Control Signal with Input Constraints');

