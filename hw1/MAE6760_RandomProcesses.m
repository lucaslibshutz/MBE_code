%% Random Processes: continuous time linear system driven by white noise
%  Quarter model of a car suspension
% 
% - Continuous-time and discrete-time state-space models
% - Time-domain simulation
% - Analytical covariance via Lyapunov equations
% - Frequency-domain PSD comparison
% 
% MAE 6760 Model Based Estimation
% Cornell University
% M Campbell
%
clear all; close all

%% Define suspension model

% Car and suspension parameters
k = 500; % spring constant (Nsec/m)
m = 325; % 1/4 mass of car (kg)
b = 400; % damper constant (Nsec^2/m)

% Continuous time (CT) state space model: 
% state = [velocity v;position x]
F=[-b/m -k/m;1 0];   %system matrix
G=[k/m-b^2/m^2;b/m]; %road input w to states
H=[0 1];             %output position
D=0;
sysSS_CT=ss(F,G,H,D);

%% Define simulation parameters including continuous and discrete white noise

% Continuous Time (CT) white noise 
Sigw_CT=0.001; %units of [w^2 * sec]

% Discrete time (DT) state space model:
dt=0.1;
sysSS_DT=c2d(sysSS_CT,dt);
Fd=sysSS_DT.A;Gd=sysSS_DT.B;Hd=sysSS_DT.C;

% DT white noise 
Sigw_DT=Sigw_CT/dt; %units of [w^2]

%% Calculate covariance (1): Simulation (time domain)

% set up simulation variables; define noises
n=10000;Tf=(n-1)*dt;t=[0:dt:Tf];
rng(0);
w=randn(n,1)*sqrtm(Sigw_CT)/sqrt(dt); %sqrt(dt) converts to DT noise

% simulate system dynamics
[z_CT,~,x_CT]=lsim(sysSS_CT,w,t);
[z_DT,~,x_DT]=lsim(sysSS_DT,w,t);

% state covariance (CT model) via sampled time domain data:
Px_CT=cov(x_CT)
% state covariance (DT model) via sampled time domain data:
Px_DT=cov(x_DT)
% output covariance (CT model): from state or data
Pz_CT=H*Px_CT*H'   %from data: Pz_CT=cov(z_CT)
% output covariance (DT model): from state or data
Pz_DT=Hd*Px_DT*Hd' %from data: Pz_DT=cov(z_DT)
% plot time domain (both CT and DT models)

figure(1);
plot(t,z_CT,'b-',t,z_DT,'r--');
ylabel('position z(t)');xlabel('time (sec)');
legend('CT position z(t)','DT position z(t)');
axis([0 100 [-1 1]*[sqrt(Pz_CT)]*5]);
grid;
PrepFigPresentation(1);

%% Calculate covariance (2): Analytical (Lyapunov-based covariance)

% state covariance (CT model) via analytical solution:
Px_CTan=lyap(F,G*Sigw_CT*G')
% state covariance (DT model) via analytical solution:
Px_DTan=dlyap(Fd,Gd*Sigw_DT*Gd')
% output covariance (CT model) via analytical solution:
Pz_CTan = H*Px_CTan*H'
% output covariance (DT model) via analytical solution:
Pz_DTan = Hd*Px_DTan*Hd'

%% Calculate covariance (3): Frequency Domain

% frequency domain transfer functions FFTs:
X=fft(x_CT)*dt; %Fourier transform
W=fft(w)*dt;    %dt is normalizing factor 
Sx_data=zeros(n,2,2);%pre-allocate for speed
Sw_data=zeros(n,2,2); 
for i=1:n
    Sx_data(i,1:2,1:2)=1/Tf*X(i,:)'*X(i,:);
    Sw_data(i,1:2,1:2)=1/Tf*W(i,:)'*W(i,:);
end
% find the power spectral density (PSD) via model
domega=2*pi/(Tf);
omega=[domega:domega:domega*(n/2)];
Sx_model=zeros(n/2,2,2);
for i=1:n/2
    Gjw=inv(j*omega(i)*eye(2)-F)*G;
    Sx_model(i,1:2,1:2)=Gjw*Sigw_CT*Gjw';  
end

% covariance of x (from model) via inverse Fourier of PSD
Rx_model=real(ifft(Sx_model)/dt);
Px_PSD_model=squeeze(Rx_model(n/2,:,:))
% covariance of x (from data) via inverse Fourier of PSD
Rx_data=real(ifft(Sx_data(1:n/2,:,:))/dt);
Px_PSD_data=squeeze(Rx_data(n/2,:,:))

% plot frequency domain: PSDs via data and model
figure(2);
loglog(omega,abs(Sx_data(1:n/2,1,1)),'b',omega,abs(Sx_model(:,1,1)),'r'); 
legend('data based','model based');
ylabel('magnitude');xlabel('frequency (rad/sec)');
PrepFigPresentation(2);


%% ---------------
%% plotting support

function PrepFigPresentation(fignum);
%
% prepares a figure for presentations
%
% Fontsize: 14
% Fontweight: bold
% LineWidth: 2
%
figure(fignum);
fig_children=get(fignum,'children'); %find all sub-plots
for i=1:length(fig_children),
set(fig_children(i),'FontSize',16);
set(fig_children(i),'FontWeight','bold');
fig_children_children=get(fig_children(i),'Children');
set(fig_children_children,'LineWidth',2);
end
end