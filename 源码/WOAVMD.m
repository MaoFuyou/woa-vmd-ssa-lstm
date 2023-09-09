function  [u, u_hat, omega,WOA_cg_curve,Target_pos]  = WOAVMD(f, tau, DC, init, tol)%Target_pos���Ż���Kֵ��Alphaֵ
% Input and Parameters:
% ---------------------
% signal  - the time domain signal (1D) to be decomposed
% alpha   - the balancing parameter of the data-fidelity constraint
% tau     - time-step of the dual ascent ( pick 0 for noise-slack )
% K       - the number of modes to be recovered
% DC      - true if the first mode is put and kept at DC (0-freq)
% init    - 0 = all omegas start at 0
%                    1 = all omegas start uniformly distributed
%                    2 = all omegas initialized randomly
% tol     - tolerance of convergence criterion; typically around 1e-6
%
% Output:
% -------
% u       - the collection of decomposed modes
% u_hat   - spectra of the modes
% omega   - estimated mode center-frequencies
%% ��������
pop = 2;%��Ⱥ����
Max_iteration = 20;%����������
lb =[500,2]; %�±߽�
ub = [2000,10];%�ϱ߽�
dim = 2; %ά��Ϊ2����alpha��K
fobj = @(x) fun(x,f,tau, DC, init, tol);
[~,Target_pos,WOA_cg_curve] = WOA(pop,Max_iteration,lb,ub,dim,fobj);%�Ż����� ��K Alpha ��ֵ
Target_pos = round(Target_pos);
%�����Ż���K��alpha����VMD�õ����
[u, u_hat, omega] = VMD(f, Target_pos(1), tau, Target_pos(2), DC, init, tol);

end