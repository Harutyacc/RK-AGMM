close all
clear
clc

% 清空日志文件
logfile = 'output_person.log';
fid = fopen(logfile, 'w');
fclose(fid);

%% 打开日志文件
logfile = 'output_person.log';
diary(logfile);

% 开始计时
elapsed_time = 0;
tic;

%% 生成符合参数的真实数据
% load('C:\Users\caoyi\OneDrive\文档\MatlabProject\KLAMM\RealEnv_Data\Direc915Test.mat');  % test_estimat_1
load('C:\Users\caoyi\OneDrive\文档\MatlabProject\KLAMM\RealEnv_Data\Person915Test.mat'); % test_estimat_2
% load('C:\Users\caoyi\OneDrive\文档\MatlabProject\KLAMM\RealEnv_Data\Vehicle915Test.mat'); % test_estimat_3

N = 10000;         
FakeCluster = 8;
TrueCluster = 4;
real_Iterations = 1000;
Iterations = real_Iterations + 1;
Max_K = 100;

% real_s,real_sigma,real_k来自场景参数的1：600帧
% real_s = strucDirec.s(1:600);
% real_sigma = strucDirec.sigma(1:600);
% real_k = strucDirec.k(1:600);     

real_s = strucMovingPerson.s(1:600);
real_sigma = strucMovingPerson.sigma(1:600);
real_k = strucMovingPerson.k(1:600);

% real_s = strucMovingVehicle.s(1:600);
% real_sigma = strucMovingVehicle.sigma(1:600);
% real_k = strucMovingVehicle.k(1:600);          

L = 600;

% min_s = min(real_s) - 5e-4;
% min_sigma = min(real_sigma) - 5e-5;
% min_k = min(real_k) - 0.5;
% max_s = max(real_s) - 5e-4;
% max_sigma = max(real_sigma) - 5e-5;
% max_k = max(real_k) - 0.5;

em_s = zeros(600,1);
em_sigma = zeros(600,1);
em_k = zeros(600,1);
em_cluster = zeros(600,1);
em_i = zeros(600,1);
em_kld = zeros(600,1);
all_kld = zeros(600,Iterations);
all_d2k = zeros(600,Iterations);
em_false = [];
false_counts = 0;

%% 600帧数据
for n = 1:600                                                          % 测试了600帧的数据
    [data, Data] = RealDataGenerator(N, TrueCluster, pi/8, real_s(n), real_sigma(n));
    % [data, Data] = DataGenerator(N, TrueCluster, pi/8, real_k(n)); % 生成对应真实参数的数据.
    data = data*600;
  
    % 构建KLD计算用hist pdf矩阵
    figure(50)
    h= histogram2(data(:,1),data(:,2),'Normalization','pdf');
    x_hist = h.XBinEdges(1:h.NumBins(1))+h.BinWidth(1)./2;
    y_hist = h.YBinEdges(1:h.NumBins(2))+h.BinWidth(2)./2;
    [X1, X2]=meshgrid(x_hist, y_hist);
    X1=X1';
    X2=X2';
    loc_hist=[X1(:) X2(:)];
    PDF_d_int = h.Values+1e-10; % 1e-10 to avoid 0
    PDF_d=reshape(PDF_d_int,h.NumBins(2)*h.NumBins(1),1);

    %% 逐帧开始训练
    K=FakeCluster;  % 以预设值K初始化k，迭代过程中会删减k
    k=K;
    KLD_d = [];
    KLD_t_judge = [];

    % 先设置初始权重
    % 权重由于会删减个数导致不成为一个标准矩阵
    w = zeros(Iterations,Max_K);
    p1 = zeros(Iterations,Max_K);
    p2 = zeros(Iterations,Max_K);
    r = zeros(Iterations,Max_K);
    KLD_t = zeros(Iterations,Max_K);
    KLD_t_score = zeros(Iterations,Max_K);
    for k1=1:k
        w(1,k1)=1/k;
    end
    u=[-2 -2]+4*rand(k,2);
    u1=u;
    % 初始协方差矩阵
    s=[ ];
    for k1=1:k
        s(:,:,k1)=[2 1;1 2];
    end

    KLDwindow = zeros(1,3);
    l = [];
    delFlag  = 1; % 1,允许拐点删除，0，不允许拐点删除。用以在过删除恢复后增加迭代概率
    decisionFlag = 0; % 延迟判决标识，拐点识别后为该标识复制，每次迭代-1，至0时启动判决。用以确保足够迭代周期
    counts_pause = 0;

    d2k = [];

    % 门限设置
    threshold = 0.1;
    recover_flag = 0;
   
    %% 帧内迭代开始
    for j=1:real_Iterations
        % E步（得到后验概率）
        counts_pause = counts_pause - 1;
        r0=zeros(N,Max_K);
        for  k1=1:k
            r0(:,k1)=w(j,k1)*mvnpdf(data,u(k1,:),s(:,:,k1));
        end
        for i=1:N
            r1(i)=0;
            for k1=1:k
                r1(i)=r0(i,k1)+r1(i); %响应度按行求和
            end
            r2(i,:)=r0(i,:)/r1(i); %响应度归一化
        end
        % M步
        for  k21=1:k
            N1(k21)=sum(r2(:,k21));
            w(j+1,k21)=N1(k21)/N;
            u(k21,:)=r2(:,k21)'*data/N1(k21);
            s2=zeros(2);
            for i1=1:N
                s1=r2(i1,k21)*(data(i1,:)-u(k21,:))'*(data(i1,:)-u(k21,:));
                s2=s2+s1;
            end
            s(:,:,k21)=s2/N1(k21);
        end


        % 
        % % 对每一组的中心都进行聚类分析一波
        % for j1=1:k
        %     p1(j+1,j1)=0;
        %     for k3=1:k
        %         p1(j+1,j1)=p1(j+1,j1)+w(j+1,k3)*mvnpdf(u(j1,:),u(k3,:),s(:,:,k3));
        %     end
        % end
        % for i=1:k
        %     p2(j+1,i)=w(j+1,i);
        % end
        % d=[];
        % for i=1:k
        %     for i1=1:k
        %         d(i,i1)=norm(u(i,:)-u(i1,:));
        %     end
        % end
        % for i=1:k
        %     d(i,i)=nan;
        %     r(j+1,i)=min(d(i,:));
        % end

        %% 构建 KLD
        p_density = PDF_d;
        q_gmm_density = zeros(size(loc_hist,1),1);
        for k1=1:k
            q_gmm_density = q_gmm_density + w(j+1,k1)*mvnpdf(loc_hist,u(k1,:),s(:,:,k1));
        end
        KLD_d(j+1) = (p_density)'*log(p_density./q_gmm_density);
        
        for k_tmp=1:k
            q_gsm_density = mvnpdf(loc_hist, u(k_tmp,:), s(:,:,k_tmp));
            KLD_t(j+1,k_tmp) = q_gsm_density'*log(q_gsm_density./p_density);
        end

        %% 构建反向KLD评价指标（乘以权重），初步理解，只适用于对称聚类（权重相等），判决指标为离散程度
        % 目标是最小化反向KL散度，但作为评分标准，反向KL散度乘以被分散后的权重，选取最小的评分即可
        % 冗余最优后反向KL散度几乎全部相等
        KLD_t_score(j+1, 1:k) = KLD_t(j+1, 1:k) .* w(j+1, 1:k);
        KLD_t_judge(j+1)=var(KLD_t_score(j+1,1:k))./mean(KLD_t_score(j+1,1:k));
        
        KLD_temp = KLDwindow(1);
        KLDwindow = [KLDwindow(2:3) KLD_d(j+1)];
        d2k(j+1) = (KLDwindow(3)-KLDwindow(2)) - (KLDwindow(2) - KLDwindow(1));

        %% 计算似然值
        l(j+1)=0;
        for k1=1:k
            l1(:,k1)=w(j+1,k1)*mvnpdf(data,u(k1,:),s(:,:,k1));
        end
        for i1=1:N
            l11(i1)=0;
            for k1=1:k
                l11(i1)=l11(i1)+l1(i1,k1);
            end
            l(j+1)=l(j+1)+log(l11(i1));
        end

        % 收敛判决，以前是通过似然值进行判别
        % 现在改为KLD
        % KLD在16qam时初期变化不大
        if  abs(KLD_d(j+1)-KLD_d(j)) <= 2e-3 % 5e-4 1e-4
            % if l(j+1)-l(j)<= 10&&l(j+1)-l(j)>0
            % KLD收敛,注意，此时我们可以判断是否时局部最优。全局最优时KLD收敛至0.
            % 在这里的错误判决有两种情况，一种是因为KLD_d的模糊性，门限太低，不能结束
            % 另一种是因为反向KLD门限太低，无法满足，也无法结束
            if KLD_d(j+1) < threshold + recover_flag*0.1 % 门限值需调试，尚可0.2 0.1 0.05 0.3
                if KLD_t_judge(j+1) > 0.15 % 门限值需调试，尚可1.5e-2 0.1 0.015 0.0001(direct)
                    % 可分，非全局最优，删除
                    disp(['收敛，非全局最优，删除聚类，j = ', num2str(j)])
                    % 启动一次核函数删除
                    [KLmin,KLindex]=min(KLD_t_score(j+1,1:k));
                    k1=k-1;
                    % 注意，这段代码原程序有bug
                    w1=w(j+1,KLindex)/k1;
                    for i=1:k
                        w2(i)=w1+w(j+1,i);
                    end
                    w2(KLindex)=0;
                    w(j+1,:) = zeros(1,Max_K);
                    u1=[];
                    s1=[];
                    i11=0;

                    for  i=1:k
                        if w2(i)>0
                            i11=i11+1;
                            w(j+1,i11)=w2(i);
                            u1(i11,:)=u(i,:);
                            s1(:,:,i11)=s(:,:,i);
                        end
                    end
                    u=u1;
                    s=s1;
                    k=k1;
                else % 不可分
                    if (abs(KLD_d(j+1)-KLD_d(j)) <= 5e-5) && (KLD_d(j+1) < 2)  %(1e-4,0.2) && (KLD_d(j+1) < 0.8)
                        disp(['收敛至全局最优，j = ', num2str(j)])
                        break; % 全局最优，退出迭代
                    end
                end
            else
                % 收敛，但是非全局最优
                % 大概率过删除，小概率局部最优。
                % 注意即使是局部最优，也可以通过增加核函数，形成扰动，跳出局部最优。
                if counts_pause <= 0
                    if (abs(KLD_d(j+1)-KLD_d(j)) <= 5e-4)  % 增加条件，防止循环 (KLD_d(j+1) <= 2) (KLD_d(j+1) < 0.8)
                        disp(['收敛，非全局最优，恢复聚类，j = ', num2str(j)])
                        delFlag = -1; %使能概率删除
                        k = k+1;
                        for k1=1:k
                            w(j+1,k1)=1/k;
                        end
                        u(k,:)=mean(u)+ 0.5 * (-1 + 2 * rand(1,2));
                        s(:,:,k)=[1 0.5;0.5 1];
                        recover_flag = recover_flag + 1;

                        disp(['恢复聚类，阈值需要上调，recover_flag = ', num2str(recover_flag)])
                    end
                end
            end
        end
            % else
            %     % 未收敛，判断拐点
            %     % 相当于表示kld的2次方，即kld的斜率变化情况
            %     if  (d2k(j)<0 && d2k(j+1)>0) && (delFlag == 1) && (KLD_temp ~= 0) && (abs(KLD_d(j+1)-KLD_d(j)) >= 0.025)  % 过0点检测拐点 && (KLD_d(j+1) <= 0.8) (KLD_d(j+1) >= 0.5) && (KLD_d(j+1) <= 1.5)
            %         if delFlag == 1 || (delFlag == -1 && KLD_d(j+1) > 2) % 出现了kld减小变缓但在j点又增加
            %             disp(['拐点,7次迭代周期后启动判决，j = ', num2str(j)])
            %             decisionFlag = 7;
            %             delFlag = 1;
            %         end
            %     end
            % end

        % Plot the converged GMM

                %% plot the converged gmm

        % hold off
        % scatter(data(:,1),data(:,2));
        % hold on
        % scatter(u(:,1),u(:,2),200,'MarkerFaceColor','r');
        % xlabel('In-phase');
        % ylabel('Quadrature');
        % figure(3)
        % bar(pr(j+1,:)');
        % xlabel('Kernels');
        % ylabel('Norm.Eucl.dis');
        % figure(4)
        % z=[];
        % for i=1:k
        %     z(i,:)=[p1(j+1,i) p2(j+1,i) r(j+1,i)];
        % end
        % scatter3(z(:,1),z(:,2),z(:,3),'r','LineWidth',2);
        % text(z(:,1),z(:,2),z(:,3),num2cell(1:k));
        % xlabel('Density');
        % ylabel('Weight');
        % zlabel('Gap');
        % figure(2)
        % x_l = 2:length(l);
        % x_kld = 2:length(KLD_d);
        % plotyy(x_l,l(2:length(l)),x_kld,KLD_d(2:length(KLD_d)))
        % legend('loglikihood','KLD(T|D)')
        % xlabel('Number of iterations');

        % Figure 1: Scatter plot of data and cluster centers
        figure(1);
        hold off
        scatter(data(:,1), data(:,2));
        hold on;
        scatter(u(:,1), u(:,2), 200, 'MarkerFaceColor', 'r');
        text(u(:,1), u(:,2), num2cell(1:length(u(:,1))));
        xlabel('In-phase');
        ylabel('Quadrature');
        % title('Data and Cluster Centers');
        % set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window

        % Figure 2: Bar plot of kernel norms
        figure(2);
        bar(KLD_t(j+1,1:k)');
        xlabel('Kernels');
        ylabel('Reverse KLD');
        % title('Kernel Norms');
        % set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window

        % Figure 3: Bar plot of density, weight, and gap
        figure(3);
        bar(KLD_t_score(j+1,1:k)');
        xlabel('Kernels');
        ylabel('Reverse KLD with Weight');
        % title('Kernel Norms');
        % set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window

        % Figure 4: Plot of log likelihood and KLD
        figure(4);
        x_l = 2:length(l);
        x_kld = 2:length(KLD_d);
        [ax, h1, h2] = plotyy(x_l, l(2:length(l)), x_kld, KLD_d(2:length(KLD_d)));
        legend('loglikelihood', 'KLD(T|D)');
        xlabel('Number of iterations');
        ylabel(ax(1), 'loglikelihood');
        ylabel(ax(2), 'KLD(T|D)');
        title('Log Likelihood and KLD');
        % set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window


    end

    % 判断用对应哪种调制方式
    if length(nonzeros(w(j+1,:))) == 4
        estimat_s = mean(sqrt(power(u(1:k, 1)./600, 2) + power(u(1:k, 2)./600, 2)));
        estimat_sigma=sqrt(mean(s(1,1,1:k)./360000 + s(2,2,1:k)./360000)/2);
        estimat_k = power(estimat_s,2)/(2*power(estimat_sigma, 2));
    else
        estimat_s = mean(sqrt(power(u(1:k, 1)/600,2)+power(u(1:k, 2)/600, 2)))/2.1177;
        estimat_sigma=sqrt(mean(s(1,1,1:k)./360000 + s(2,2,1:k)./360000)/2);
        estimat_k = power(estimat_s,2)/(2*power(estimat_sigma, 2));
    end

    if (j >= real_Iterations)
        if (k ~= 4)
            disp("******************第" + num2str(n) + "帧估计失败 | 聚类数目为" + k +"!******************");
        else
            disp("******************第" + num2str(n) + "帧估计待定 | 聚类数目为" + k + "需根据ekld判定!******************");
        end
        clus.flase(n) = 1;
        false_counts = false_counts + 1;
        em_false(false_counts) = n;

        % 保存此时的星座图，以便调试
        figure(1000+false_counts)
        hold off
        scatter(data(:,1),data(:,2));
        hold on
        scatter(u(:,1),u(:,2),200,'MarkerFaceColor','r');
        xlabel('In-phase');
        ylabel('Quadrature');

        folder = 'wrong_folder\vehicle'; % 设置文件夹名称
        if ~exist(folder, 'dir')
            mkdir(folder); % 如果文件夹不存在，则创建
        end
        filename = fullfile(folder, ['Frame_' num2str(n) '.fig']); % 保存图形为 .fig 格式
        savefig(filename);
    else
        disp("******************第" + num2str(n) + "帧估计成功!******************");
    end


    em_s(n) = estimat_s;
    em_sigma(n) = estimat_sigma;
    em_k(n) = estimat_k;
    em_cluster(n) = k;
    em_i(n) = j;
    em_kld(n) = KLD_d(j+1);
    all_kld(n,1:length(KLD_d)) = KLD_d;
    all_d2k(n,1:length(d2k)) = d2k;

    disp("real_s = " + real_s(n) + " | estimat_s = " + em_s(n));
    disp("real_sigma = " + real_sigma(n) + " | estimat_sigma = " + em_sigma(n));
    disp("real_k = " + real_k(n) + " | estimat_k = " + em_k(n));
    disp("TrueCluster = " + 4 + " | estimat_cluster = " + em_cluster(n));
    disp("Iterations = " + em_i(n) + " | empirical_KLD = " + em_kld(n));
    if (~isempty(em_false))
        disp("截至此帧，已有" + length(em_false) + "帧数据估计出错或待检验");
        disp("出错帧的位置索引为:{" + em_false + "}");
    end
    disp("************************************************")

    figure(5)
    hold off
    scatter(data(:,1),data(:,2));
    hold on
    scatter(u(:,1),u(:,2),50,'MarkerFaceColor','r');
    x1=-2:0.1:2;
    x2=-2:0.1:2;
    [X1, X2]=meshgrid(x1, x2);
    X=[X1(:) X2(:)];
    for k1=1:k
        y1=w(j+1,k1)*mvnpdf(X,u(k1,:),s(:,:,k1));
        y1=reshape(y1,length(x2),length(x1));
        contour(x1, x2, y1);
    end
    xlabel('I');
    ylabel('Q');

    figure(6)
    subplot 414
    x_l = 2:length(l);
    x_kld = 2:length(KLD_d);
    plotyy(x_l,l(2:length(l)),x_kld,KLD_d(2:length(KLD_d)))
    legend('loglikihood','KLD(T|D)')
    xlabel('迭代次数');
    subplot 411
    plot(p1(1:j+1));
    ylabel('密度');
    subplot 412
    plot(p2(1:j+1));
    ylabel('权重');
    subplot 413
    plot(r(1:j+1));
    ylabel('最小间距');

    %% plot the pr metric
    figure(7)
    x_l = 2:length(l);
    x_kld = 2:length(KLD_d);
    [hAx,hLine1,hLine2]=plotyy(x_l,l(2:length(l)),x_kld,KLD_d(2:length(KLD_d)));
    xlabel('Number of iterations');
    ylabel(hAx(1),'Loglikihood');
    ylabel(hAx(2),'KLD(T|D)','FontName','Times New Roman','FontSize',30);

    %% 绘制参数拟合图
    % s和sigma拟合
    figure(21)
    % 左轴
    yyaxis left
    x_frame = 1:n;
    plot(x_frame, real_s(1:n), '-', 'Color', '#8ECFC9', 'Marker', 'none');  % 绘制真实s
    hold on
    plot(x_frame, em_s(1:n), '-', 'Color', '#FFBE7A', 'Marker', 'none');  % 绘制估计s
    ylabel('s');
    % ylim([min_s, max_s]);  % 设置左轴范围
    % yticks(min_s:5e-4:max_s);  % 设置左轴刻度
    % 右轴
    yyaxis right
    plot(x_frame, real_sigma(1:n), '-', 'Color', '#FA7F6F', 'Marker', 'none');  % 绘制真实sigma
    hold on
    plot(x_frame, em_sigma(1:n), '-', 'Color', '#82B0D2', 'Marker', 'none');  % 绘制估计sigma
    ylabel('sigma');
    % ylim([min_sigma, max_sigma]);  % 设置右轴范围为真实sigma的最大值
    % yticks(min_sigma:5e-5:max_sigma);  % 设置右轴刻度
    xlabel('frame');
    % 图例
    legend('s-True','s-AMM','sigma-True','sigma-AMM');

    % k拟合
    figure(22)
    plot(x_frame, real_k(1:n), '-', 'Color', '#BEB8DC', 'Marker', 'none');  % 绘制真实k
    hold on
    plot(x_frame, em_k(1:n), '-', 'Color', '#E7DAD2', 'Marker', 'none');  % 绘制估计k
    ylabel('k');
    % ylim([min_k, max_k]);
    % yticks(min_k:1:max_k);
    % 图例
    legend('k-True', 'k-AMM');
    xlabel('frame');

    %% 计时和保存
    elapsed_time_single = toc - elapsed_time;
    disp("此帧用时为" + num2str(elapsed_time_single) + " seconds！");
    disp("************************************************");
    save('test_estimat_1.mat') % 保存当前所有变量到文件
    elapsed_time = toc;

end

disp("总用时为" + num2str(elapsed_time) + " seconds！");
disp("已估计完所有帧！！！！！已保存所有变量到文件中，保存所有输出到日志中可供查询！");
diary off;

