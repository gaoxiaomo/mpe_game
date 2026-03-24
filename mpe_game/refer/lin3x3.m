close all;
clc;
clear;
global Wo;

R = 1*[1 0;0 1];
Q=1*[1 0 0;0 1 0;0 0 1];

CNT=13;
Ts=.2;
LB=-1.2;
A1=3;%3
A2=20;%20


index=0;
for i=1:CNT
    x1=LB+(i-1)*Ts*1;
    for j=1:CNT
       x2=LB+(j-1)*Ts*1;            
       for k=1:CNT            
          x3=LB+(k-1)*Ts*1; 
          dPHI=[ 	2*x1            0               0
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
                 
           f = [2*x1+x2+x3;
               x1-x2;
               x3];

		     g = [ 0  0;
                   0  1;
                   1  0];

      
        	  u = [A1*tanh((1/A1)*(-8.3056*x1-2.2827*x2-4.6607*x3));
                A2*tanh((1/A2)*(-8.5707*x1-2.7323*x2-2.2827*x3))];
             
             u = [-8.3056*x1-2.2827*x2-4.6607*x3;
                -8.5707*x1-2.7323*x2-2.2827*x3];
        
                u1=u(1);u2=u(2);
     
                if u1>=A1*(1-1e-14)
                    u1=A1*(1-1e-14);
                elseif u1<=-A1*(1-1e-14)
                    u1=-A1*(1-1e-14);
                end
                
                if u2>=A2*(1-1e-14)
                    u2=A2*(1-1e-14);
                elseif u2<=-A2*(1-1e-14)
                    u2=-A2*(1-1e-14);
                end
                u=[u1;u2];
             
        	  Z       = dPHI*f+dPHI*g*u;            
        	  index=index+1;     
              y(index,1)  = -[x1 x2 x3]*Q*[x1;x2;x3] -2*A1*R(1,1)*(u1*atanh(u1/A1)+0.5*A1*log(1.0-(u1/A1)^2))-2*A2*R(2,2)*(u2*atanh(u2/A2)+0.5*A2*log(1.0-(u2/A2)^2));
           X(index,:)=Z';
        end         
                     
     end
end
W=X\y;
clc;
Wo=W
pause

for n=1:20
    index=0;
    for i=1:CNT
        x1=LB+(i-1)*Ts*1;
        for j=1:CNT
           x2=LB+(j-1)*Ts*1;            
           for k=1:CNT
              x3=LB+(k-1)*Ts*1; 
              dPHI=[ 2*x1            0               0
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
                     0               3*x2^2*x3       x2^3];              
     
		     f = [2*x1+x2+x3;
               x1-x2;
               x3];

		     g = [ 0  0;
                   0  1;
                   1  0];
           
			      U = -0.5*inv(R)*g'*dPHI'*Wo;
               u = [A1*tanh(1/A1*U(1));
                  A2*tanh(1/A2*U(2))];
               u1=u(1);u2=u(2);


               if u1>=A1*(1-1e-14)
                    u1=A1*(1-1e-14);
                elseif u1<=-A1*(1-1e-14)
                    u1=-A1*(1-1e-14);
                end
                
                if u2>=A2*(1-1e-14)
                    u2=A2*(1-1e-14);
                elseif u2<=-A2*(1-1e-14)
                    u2=-A2*(1-1e-14);
                end
                
                Z       = dPHI*f+dPHI*g*u;            
     			index=index+1;                
              y(index,1)  = -[x1 x2 x3]*Q*[x1;x2;x3] -2*A1*R(1,1)*(u1*atanh(u1/A1)+0.5*A1*log(1.0-(u1/A1)^2))-2*A2*R(2,2)*(u2*atanh(u2/A2)+0.5*A2*log(1.0-(u2/A2)^2));
        
            X(index,:)=Z';
         end
      end
   end     
W=X\y;
clc;
n
Wo=W
pause
end
% [k,s,e]=lqr([2 1 1;1 -1 0;0 0 1],[0 0;0 1;1 0],Q,R)