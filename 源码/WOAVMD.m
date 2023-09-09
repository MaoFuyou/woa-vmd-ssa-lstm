function  [u, u_hat, omega,WOA_cg_curve,Target_pos]  = WOAVMD(f, tau, DC, init, tol)%Target_pos：优化的K值和Alpha值
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
%% 参数设置
pop = 2;%种群数量
Max_iteration = 20;%最大迭代次数
lb =[500,2]; %下边界
ub = [2000,10];%上边界
dim = 2; %维度为2，即alpha，K
fobj = @(x) fun(x,f,tau, DC, init, tol);
[~,Target_pos,WOA_cg_curve] = WOA(pop,Max_iteration,lb,ub,dim,fobj);%优化函数 求K Alpha 熵值
Target_pos = round(Target_pos);
%利用优化后K，alpha带入VMD得到结果
[u, u_hat, omega] = VMD(f, Target_pos(1), tau, Target_pos(2), DC, init, tol);

end