tablemenu_=[100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,232];
% 获取当前时间戳
dateTimeStamp = datestr(now, 'mm-dd_HH-MM-SS');
dataFolderPath = fullfile('D:\Desktop', ['data' dateTimeStamp]);
if ~exist(dataFolderPath, 'dir')
    mkdir(dataFolderPath);
end
% 日志文件
logFile = fopen(fullfile(dataFolderPath, sprintf('output_log_%s.txt', dateTimeStamp)), 'a'); % 使用 'a' 模式以追加内容

% tablemenu_=[103,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,232];
%tablemenu_=[209,119,121,122,123,124,200,201,202,203,205,207,208,209,210,232];
for tablemenu_i = 1:length(tablemenu_)
    clc; clearvars -except tablemenu_ tablemenu_i logFile dataFolderPath;
    %------ LOAD DATA USING WFDB TOOLBOX -------------------------------------
    stringname=num2str(tablemenu_(tablemenu_i));
    % 设置路径和文件名
    dataPath = ['G:\tunnel\1.0.0\', stringname, '\'];
    cd(dataPath);
    % 使用 wfdbdesc 读取头文件信息
    siginfo = wfdbdesc(stringname);
    % 显示所有字段名
    %   disp(fieldnames(siginfo));
    % 显示信号通道数量和每个通道的导联信息
    signum = length(siginfo);
    fprintf(logFile, 'Record %s has %d signal(s):\n', stringname, length(siginfo));
    for i = 1:signum
        fprintf(logFile, 'Signal %d: %s, Gain: %s, ADC Resolution: %s \n', i, siginfo(i).Description, siginfo(i).Gain, siginfo(i).AdcResolution);
    end
    

    % 使用 rdsamp 和 rdann 加载数据和注释
    [tm, sig] = rdsamp(stringname);
    [ATRTIME, ANNOT] = rdann(stringname, 'atr');
    sfreq = 360;
    points=length(tm);

    % %%
    % % 计算时间向量
    % TIME = (0:points-1) / sfreq;
    % 
    % % 绘制数据
    % figure;
    % hold on;
    % grid on;
    % 
    % % 绘制所有信号通道
    % for i = 1:signum
    %     plot(TIME, tm(:,i), 'DisplayName', sprintf('Signal %d: %s', i, siginfo(i).Description));
    % end
    % 
    % % 添加注释
    % % for k = 1:length(ANNOT)
    % %     text(ATRTIME(k)/Fs, tm(round(ATRTIME(k)),1), num2str(ANNOT(k)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'BackgroundColor', 'w');
    % % end
    % 
    % % 设定图形属性
    % xlabel('Time (s)');
    % ylabel('Voltage (mV)');
    % title(['ECG Signals from Record ', stringname]);
    % legend show;
    % xlim([TIME(1), TIME(end)]);
    % fprintf(1,'\\n$> DISPLAYING DATA FINISHED \n');
    % 

    ECGsignalM1=tm(:,1);
    %ECGTsignalM2=tm(:,2);
    cd(dataFolderPath);

    clear Fs i sigClass signum tm
    %%
    %去除噪声和基线漂移
    level=8; wavename='bior2.6';
    %ecgdata=ECGsignalM1+ECGTsignalM2;
    ecgdata=ECGsignalM1;
    %ecgdata=ECGTsignalM2;
    zeroDetailLevels = 2; % 你想要置零的最高频率细节系数的数量
            DenoisingSignal = WaveletDenoiseFlexible(ecgdata, level, wavename, zeroDetailLevels);


    %%
    %           initiate
    close all;
    %clear;clc;
    sig=DenoisingSignal;
    %sig=load('ecg_60hz_200.dat');
    N=length(sig);
    fs=360;
    t=[0:N-1]/fs;
    %figure(1);subplot(4,2,1);plot(sig)
    %title('Original Signal')


    %%
    %           Low Pass Filter

    b=1/32*[1 0 0 0 0 0 -2 0 0 0 0 0 1];
    a=[1 -2 1];
    sigL=filter(b,a,sig);
    %subplot(4,2,3);plot(sigL)
    %title('Low Pass Filter')
    %subplot(4,2,4);zplane(b,a)

    %%
    %           High Pass Filter

    b=[-1/32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1/32];
    a=[1 -1];
    sigH=filter(b,a,sigL);
    %subplot(4,2,5);plot(sigH)
    %title('High Pass Filter')
    %subplot(4,2,6);zplane(b,a)

    %%
    %          Derivative Base Filter

    b=[1/4 1/8 0 -1/8 -1/4];
    a=[1];
    sigD=filter(b,a,sigH);
    %subplot(4,2,7);plot(sigD)
    %title('Derivative Base Filter')
    %subplot(4,2,8);zplane(b,a)

    %%
    %      be tavane 2 miresanim
    sigD2=sigD.^2;

    %%
    %      normalization
    signorm=sigD2/max(abs(sigD2));

    %%

    h=ones(1,31)/31;
    sigAV=conv(signorm,h);
    sigAV=sigAV(15+[1:N]);
    sigAV=sigAV/max(abs(sigAV));
    %figure(2);plot(sigAV)
    %title('Moving Average filter')

    %%
    treshold=mean(sigAV);
    P_G= (sigAV>0.01);
    %figure(3);plot(P_G)
    %title('treshold Signal')
    %figure;plot(sig)
    %%
    difsig=diff(P_G);
    left=find(difsig==1);
    raight=find(difsig==-1);

    %%
    %      run cancel delay
    %      6 sample delay because of LowPass filtering
    %      16 sample delay because of HighPass filtering
    left=left-(6+16);
    raight=raight-(6+16);
    %%
%        %临时绘图
%         % 定义颜色
%         colors = {[0.1, 0.6, 0.2], [0.2, 0.2, 0.6], [0.6, 0.2, 0.2], [0.7, 0.7, 0.2]}; % 颜色1，颜色2，颜色3，颜色4
%         darkColors = {[0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0], [0.6, 0.6, 0]}; % 深色版本
%         tau = 10;
%         % 组的数据
%         data = [394, 422, 518, 587, 633, 670, 698, 716;
% 828, 862, 910, 1038, 1078, 1094, 1124, 1139;
% 1264, 1293, 1361, 1456, 1490, 1529, 1557, 1576;
% 1702, 1742, 1792, 1903, 1933, 1962, 1994, 2011;];
%         startpoint = [265];
%         T_starts = [startpoint(tablemenu_i), data(4*(tablemenu_i)-3,8), data(4*(tablemenu_i)-2,8), data(4*(tablemenu_i)-1,8)]; % T起始位置，组1的开始和之后每组的T起始等于上一组的S
% 
%         % 绘制波形图
%         clf;
%         figure;
%         hold on;
%         for i = 1:4
%             % 获取每组数据的起始和结束索引
%             dataselect = 4*(tablemenu_i-1)+i;
%             T_start = T_starts(i);
%             T_peak = data(dataselect, 1);
%             T_end = data(dataselect, 2);
%             U_peak = data(dataselect, 3);
%             U_end = data(dataselect, 4);
%             P_peak = data(dataselect, 5);
%             Q = data(dataselect, 6);
%             R = data(dataselect, 7);
%             S = data(dataselect, 8);
%             U_start = T_end;
%             P_start = U_end;
%             subplot(2,1,1);
%             hold on;
%             % 绘制每个段，并为每个峰值或点标记
%             plot(T_start:T_end, sigL(T_start:T_end), 'Color', colors{1});
%             plot(U_start:U_end, sigL(U_start:U_end), 'Color', colors{2});
%             plot(P_start:Q, sigL(P_start:Q), 'Color', colors{3});
%             plot(Q:S, sigL(Q:S), 'Color', colors{4});
% 
%             % 标记峰值和QRS点
%             plot(T_peak, sigL(T_peak), 'o', 'MarkerEdgeColor', darkColors{1}, 'MarkerFaceColor', darkColors{1}, 'MarkerSize', 2);
%             plot(U_peak, sigL(U_peak), 'o', 'MarkerEdgeColor', darkColors{2}, 'MarkerFaceColor', darkColors{2}, 'MarkerSize', 2);
%             plot(P_peak, sigL(P_peak), 'o', 'MarkerEdgeColor', darkColors{3}, 'MarkerFaceColor', darkColors{3}, 'MarkerSize', 2);
%             plot(Q, sigL(Q), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 2); % Q点
%             plot(R, sigL(R), 'o', 'MarkerEdgeColor', 'm', 'MarkerFaceColor', 'm', 'MarkerSize', 2); % R点
%             plot(S, sigL(S), 'o', 'MarkerEdgeColor', 'c', 'MarkerFaceColor', 'c', 'MarkerSize', 2); % S点
% 
%             %title('Electrocardiogram (ECG) Waveform');
%             xlabel('Time');
%             ylabel('Amplitude');
% 
% 
%             subplot(2,1,2);
%             hold on;    
%             % T段
%             plot3(sigL(T_start:T_end), sigL(T_start+tau:T_end+tau), sigL(T_start+2*tau:T_end+2*tau), '-', 'Color', colors{1});
%             % U段
%             plot3(sigL(T_end:U_end), sigL(T_end+tau:U_end+tau), sigL(T_end+2*tau:U_end+2*tau), '-', 'Color', colors{2});
%             % P段
%             plot3(sigL(U_end:Q), sigL(U_end+tau:Q+tau), sigL(U_end+2*tau:Q+2*tau), '-', 'Color', colors{3});
%             % QRS段
%             plot3(sigL(Q:S), sigL(Q+tau:S+tau), sigL(Q+2*tau:S+2*tau), '-', 'Color', colors{4});
% 
%             % 标记T、U、P峰值和QRS点
%             plot3(sigL(T_peak), sigL(T_peak+tau), sigL(T_peak+2*tau), 'o', 'MarkerEdgeColor', darkColors{1}, 'MarkerFaceColor', darkColors{1}, 'MarkerSize', 2);
%             plot3(sigL(U_peak), sigL(U_peak+tau), sigL(U_peak+2*tau), 'o', 'MarkerEdgeColor', darkColors{2}, 'MarkerFaceColor', darkColors{2}, 'MarkerSize', 2);
%             plot3(sigL(P_peak), sigL(P_peak+tau), sigL(P_peak+2*tau), 'o', 'MarkerEdgeColor', darkColors{3}, 'MarkerFaceColor', darkColors{3}, 'MarkerSize', 2);
%             plot3(sigL(Q), sigL(Q+tau), sigL(Q+2*tau), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 2);
%             plot3(sigL(R), sigL(R+tau), sigL(R+2*tau), 'o', 'MarkerEdgeColor', 'm', 'MarkerFaceColor', 'm', 'MarkerSize', 2);
%             plot3(sigL(S), sigL(S+tau), sigL(S+2*tau), 'o', 'MarkerEdgeColor', 'c', 'MarkerFaceColor', 'c', 'MarkerSize', 2);            
%             %title('Phase Space Trajectory of ECG Signal');
%             xlabel('x(t)');
%             ylabel('x(t+τ)');
%             zlabel('x(t+2τ)');
%             grid on;
%             axis tight;
%             view([1, 0.5, 1]); % 设置特定的视角
%             hold on;
%         end
%             % 构建文件路径和名称
%     figName = fullfile('D:\Desktop\114514', [num2str(tablemenu_(tablemenu_i)) '-m2.fig']);
%     pngName = fullfile('D:\Desktop\114514', [num2str(tablemenu_(tablemenu_i)) '-m2.png']);
% 
%     % 保存.fig文件
%     savefig(figName);
%     exportgraphics(gcf, pngName, 'Resolution', 300);

clear a b DenoisingSignal difsig ecgdatSa ECGsignalM1 sfreq h P_G left level N raight sig sigAV sigD sigD2 sigH signorm treshold zeroDetailLevels
    %%
    %检测相图
    partplot = 2000;
    range_A = 1.5; 
    range_D = 1.8; 
    tau = 10;
    Lm = 0.03;
    %C1 = false;
    C2 = true;
    start_amplitude = 0;
    end_amplitude = Inf;
    start_position = 0;
    end_position = 0;
    firstPosChange = 0; % 记录第一次从负变正的位置
    max_value = -Inf;
    segments = cell(0, 7);
    % 初始化导数的最大值和最小值
    derivative_max = 0;
    derivative_min = 0;
    A = 16; % 定义重采样的点数

    sizesegment = 0;
    %clusterCenters = {}; % 存储每个聚类中心的曲线段
    clusterLabels = cell(0, 1); % 每个单元存储被分配到该聚类的segments的索引
    
    maxClusters = 200; % 类别上限
    minSegmentsInCluster = 5; % 类别中最少的曲线段数量
    distanceThreshold = 0.14; % 距离阈值
    labels=[];
    clusterCenterLabel=[];
    numrecognized = 6;
    colors = cell(0,1);

    for i = 1:length(sigL) - tau - 1
        % 计算当前导数并更新最大值和最小值
        current_derivative = sigL(i+tau) - sigL(i);
        current_amplitude = sigL(i+tau) + sigL(i);
        if current_derivative > 0 && sigL(i+1) > sigL(i+1+tau)  % 由正变负
            if end_position ~= 0 && end_position ~= start_position && C2 && (current_amplitude - end_amplitude) > Lm
                %range = derivative_max - derivative_min;  % 计算范围

                X = sigL(start_position:end_position);
                Y = sigL(start_position+tau:end_position+tau);
                % 计算凸包
                %k = convhull(X, Y);
                % 使用凸包顶点进行均匀采样
                %pt = interparc(A, X(k), Y(k), 'linear');
                pt = interparc(A, X, Y, 'linear');
                % 计算range_amplitude
                range_amplitude = max(max_value - end_amplitude, max_value - start_amplitude);
                pt21 = pt(:,2) - pt(:,1);
                pt12 = pt(:,1) + pt(:,2);
                boolselector = pt12 > 0.22;
                derivative_max=max([pt21(boolselector);0]);
                derivative_min=min([pt21(boolselector);0]);           
                range = derivative_max - derivative_min;  % 计算范围

                % 计算shrink
                shrink = min([1, range_A / range_amplitude, range_D / range]);

                % 对pt整体进行放缩
                %pt(:,1) = (pt(:,1) - mean(pt(:,1))) * shrink; % 对x坐标放缩
                %pt(:,2) = (pt(:,2) - mean(pt(:,2))) * shrink; % 对y坐标放缩
                 % 对pt整体进行放缩
%                pt(:,1) = pt(:,1) * shrink; % 对x坐标放缩
%                pt(:,2) = pt(:,2) * shrink; % 对y坐标放缩
                % 对pt整体进行放缩
                
                pt(:,1) = (pt12 - end_amplitude ) * shrink; % 对x坐标旋转并放缩
                pt(:,2) = (pt21 - ( derivative_max + derivative_min )/2 ) * shrink; % 对y坐标旋转并放缩

                

                % 存储重采样后的点到segments
                newSegment = {start_position, end_position, firstPosChange, range_amplitude, range, sigL(firstPosChange + 5), pt};
                segments(end+1, :) = newSegment;
                sizesegment = sizesegment+1;
                % 计算新曲线段与现有聚类中心的距离
                minDistance = inf;
                closestCluster = 0;
                lengthclusters = length(clusterCenterLabel);
                for j = 1:lengthclusters
                    distance = DiscreteFrechetDist(pt, segments{clusterCenterLabel(j),7});
                    if distance < minDistance
                        minDistance = distance;
                        closestCluster = j;
                    end
                end
                if closestCluster > 0
                    % 获取closestCluster类中的所有段索引
                    clusterMembers = clusterLabels{closestCluster};
                    numSamples = min(10, length(clusterMembers) - 1); % 抽取样本数量，最多10个，至少不包含最后一个
                    
                    % 随机抽取numSamples个索引
                    if numSamples > 0
                        sampledIndices = randperm(length(clusterMembers) - 1, numSamples);
                        sampledIndices = clusterMembers(sampledIndices);
                    else
                        sampledIndices = [];
                    end
                    
                    % 加入最后一个索引
                    sampledIndices = [sampledIndices, clusterMembers(end)];
                    
                    % 计算与这些抽取的段的距离
                    distances = arrayfun(@(idx) DiscreteFrechetDist(pt, segments{idx, 7}), sampledIndices);
                    
                    % 取最小距离
                    [minDistance, minIdx] = min(distances);
                    
                    % 取距离平均值最接近minDistance的段更新聚类中心
                    [~, closestToMeanIdx] = min(abs(mean(distances) - distances));
                    newCenterIndex = sampledIndices(closestToMeanIdx);
                    
                end
                % 判断是否需要新开一个类别
                if minDistance > distanceThreshold * max(range,range_amplitude)+0.09 || closestCluster == 0
                    if lengthclusters < maxClusters
                        %clusterCenters{end+1} = pt;
                        % 初始化这个新类别的索引列表，加入当前段的索引
                        clusterLabels{end+1} = sizesegment; % 假设这是在将当前段添加到segments之前
                        label =numrecognized+lengthclusters+1; % 获取当前段落的label
                        labels(sizesegment)=label;
                        clusterCenterLabel(lengthclusters+1)=sizesegment;
                    %else
                    %    labels(sizesegment)=-1;
                        
                    else
                        labels(sizesegment)=closestCluster;
                    end
                    
                    % 根据label的值选择颜色
                    if label == -1
                        color = [1, 0, 0]; % 红色
                    elseif label == 1
                        color = [0, 1, 0]; % 绿色
                    elseif label == 2
                        color = [0, 0, 1]; % 蓝色
                    elseif label == 3
                        color = [0.6, 0.6, 0]; % 黄色
                    elseif label == 4
                        color = [0, 0.75, 0.75]; % 青色
                    elseif label == 5
                        color = [0.75, 0, 0.75]; % 紫色
                    else
                        % label大于3时，颜色从灰色变为接近黑色

                        color = [1*rand(),1*rand(),1*rand()];
                    end
                    colors{label+2}= color;
                else
                    % 更新聚类中心
                    %clusterCenters{closestCluster} = segments{newCenterIndex, 6};
                    clusterCenterLabel(closestCluster) = newCenterIndex;
                    clusterLabels{closestCluster}(end+1) = sizesegment; % 添加当前段的索引到对应的聚类
                    labels(sizesegment)=closestCluster;
                end
               
                %
                % 绘制重采样后的点
                %
                % plot(X, Y, '-');
                % hold on;
                % xlabel('sigL(i)');
                % ylabel('sigL(i+tau)');
                % title(sprintf('ECG Segment %d Takens Embedding', i));
                % plot(pt(:,1), pt(:,2), '-');
                % grid on;
                % axis([-1 1.5 -1 1.5]);
                % pause(0.01);
                % hold on;
                % 检查segments的长度，确定是否绘制图形

                % if length(segments) >= partplot
                %     clf; % 清除当前图形窗口的内容
                %
                %     % 在循环内绘制线和点，并染色
                %     for p = max(1, length(segments)-partplot+1):length(segments)
                %         % 提取当前段落的起始和结束位置
                %         startIdx = segments{p, 1};
                %         endIdx = segments{p, 2};
                %         posChange = segments{p, 5} + 5;
                %         ptx=segments{p, 6}(:,1);
                %         pty=segments{p, 6}(:,2);
                %         sigValue = sigL(posChange);
                %         label = labels(p);
                %         color = colors{label+2};
                %
                %
                %
                %         % 绘制这一段时间的波形和三维相空间图
                %         % 波形图
                %         subplot(2,1,1);
                %         plot(startIdx:endIdx, sigL(startIdx:endIdx), '-', 'Color', color); % 为线段染色
                %         hold on; % 保持绘图，准备下一个绘图
                %         plot(posChange, sigL(posChange), 'o', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerSize', 3); % 为点染色
                %
                %         % 三维相空间图
                %         subplot(2,1,2);
                %
                %         plot(ptx, pty, '-', 'Color', color);
                %         %plot3(sigL(startIdx:endIdx), sigL(startIdx+tau:endIdx+tau), sigL(startIdx+tau*2:endIdx+tau*2), '-', 'Color', color); % 为线段染色
                %         hold on; % 保持绘图，准备下一个绘图
                %         %plot3(sigL(posChange), sigL(posChange+tau), sigL(posChange+2*tau), 'o', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerSize', 3); % 为点染色
                %     end
                %
                %     % 设置波形图和三维相空间图的共通属性
                %     subplot(2,1,1);
                %     title('波形图');
                %     xlabel('Time');
                %     ylabel('Amplitude');
                %     subplot(2,1,2);
                %     title('相空间图');
                %     xlabel('sigL(i)');
                %     ylabel(['sigL(i+', num2str(tau),')']);
                %     %zlabel(['sigL(i+', num2str(2*tau),')']);
                %     %view([1, 0.5, 1]);
                %     axis equal;
                %     grid on;
                %     axis([-1 1.5 -1 1.5]);
                %     axis tight;
                %
                %     hold on; % 解除绘图保持状态
                %     pause(0.01); % 短暂暂停以便观察
                % end

                start_position = end_position;
                start_amplitude = end_amplitude;
                end_position = 0;
                end_amplitude = Inf;
                derivative_max = 0;
                derivative_min = 0;
                max_value = current_amplitude;
                firstPosChange = i;
                C2 = false;

            end
            if current_amplitude > max_value
                max_value = current_amplitude;
                firstPosChange = i;
            end

        elseif current_derivative < 0 && sigL(i+1) < sigL(i+1+tau)  % 由负变正
            if max_value ~= -Inf && max_value - current_amplitude > Lm
                C2 = true;
            end
            if start_position == 0
                start_position = i;
                start_amplitude = current_amplitude;
                firstPosChange = i; % 记录第一次从负变正的位置
            elseif  C2 && end_amplitude > current_amplitude
                end_amplitude = current_amplitude;
                end_position = i;
            end
        end
    end
    clear boolselector C2 closestCluster closestToMeanIdx clusterMembers color current_amplitude;
    clear current_derivative derivative_min derivative_max distance distances end_position end_amplitude ;
    clear range_amplitude shrink sampledIndices start_amplitude start_position X Y;
    clear firstPosChange i j label max_value minDistance minidx newCenterIndex newSegment pt pt12 pt21 range ;
    %%
    %二次处理逻辑

% 假设segments、clusterLabels、clusterCenters已经根据上述逻辑填充完毕
sizesegment = length(segments); % 假设sizesegment已正确计算
h = length(segments); % 曲线段的数量
partplot=h;

% 对clusterLabels中的每个元素长度进行排序，并保留排序前的索引
[sortedLengths, sortedIndices] = sort(cellfun(@length, clusterLabels), 'descend');

% 根据排序结果生成QRSlabel，它包含对应clusterCenterLabel中位置的排序

% 计算总段数的5%
thresholdLength = sum(sortedLengths) * 0.02;

% 累计长度，直到达到5%的总段数
accumLength = 0;
idxToRemove = [];
% 根据条件进一步筛选QRSlabel
QRSlabel = [];
Tlabel = [];
for i = lengthclusters:-1:1
    accumLength = accumLength + sortedLengths(i);
    if accumLength >= thresholdLength
        break;
    end
    idxToRemove = [idxToRemove, i]; % 记录需要剔除的索引
end

% 剔除这些索引对应的聚类，并在labels中标记为-1
for idx = idxToRemove
    actualIdx = sortedIndices(idx); % 实际需要剔除的聚类在clusterLabels中的索引
    if length(clusterLabels{actualIdx}) <= 20
        for segIdx = clusterLabels{actualIdx}
            labels(segIdx) = -1; % 标记为-1
        end
    end
    %sortedIndices(idx) = []; % 从QRSlabel中移除
end

for labelIdx = 1:lengthclusters
    clusterCenterLabelIdx = clusterCenterLabel(labelIdx);
    watch1 = segments{clusterCenterLabelIdx,6};
    watch2 = segments{clusterCenterLabelIdx,4};
    watch3 = segments{clusterCenterLabelIdx,5};
    watch4 = segments{clusterCenterLabelIdx,7}(1,1);
    %shrink = min([1, range_A / watch2, range_D / watch3]);
    if watch1 > 0.4 && watch3 >0.5 && (watch3/watch2) > 0.5  % 满足特定条件
        QRSlabel = [QRSlabel, labelIdx]; % 保留这个聚类中心标签
        for segIdx = clusterLabels{labelIdx}
            labels(segIdx) = 0; % 标记为0(QRS)
        end
    elseif watch4 < -0.15
        Tlabel = [Tlabel, labelIdx];
        for segIdx = clusterLabels{labelIdx}
            labels(segIdx) = 1; % 标记为1(T)
        end
    end
end
clear i A segIdx wavename thresholdLength actualIdx clusterCenterLabelIdx idx idxToRemove labelIdx Lm minSegmentsInCluster t shrink sortedLengths watch1 watch2 watch3 watch4;

% 假定已有labels数组和sigL信号数组
current_state = 'escapeT';
% 预先计算所有 segments 的振幅
allAmplitudes = cell2mat(segments(:, 6));
current_position = 5;
direction = 1;  % 初始默认向右
segmentCount = 0;
offset = 0;
LabelToChange = 1;
%-1：噪声信号
%0：QRS波
%1：T，或者是心跳补动时候很像T的形状。
%2：补动T
%3：U
%4：P
%5：补动P(Q)（心跳补动时候很像P或QRS的形状）
%6：整波PT
while current_position <= sizesegment
    currentlabel=labels(current_position);
    switch current_state
        case 'FindT'
            if currentlabel == -1
            elseif currentlabel == 0  % 发现Q波
                if segments{current_position,4} > 0.3
                    current_state = 'FoundT';
                    segmentCount = 0;  % 重置段计数
                    direction = -1;  % 改变方向向左
                end
            elseif currentlabel == 1  % 在发现Q波之前发现T波
                if segments{current_position-1,4} > 0.3
                    current_state = 'escapeQ';
                    segmentCount = 1;  % 重置段计数
                    labels(current_position-1) = 5;  % 标记为补动P
                    direction = -1;  % 改变方向向左
                end
            end
        case 'escapeT'
            % if segments{current_position,4} < 0.5
            %     labels(current_position) = -1;
            % end
            if currentlabel == -1||currentlabel == 1
            else current_state = 'FindT';
                current_position = current_position -1;
            end

        case 'escapeQ'
            if (currentlabel == 0 || currentlabel == 5) && segments{current_position,4} > 0.3 || currentlabel == -1
            else
                if direction == -1
                    current_state = 'FoundT';
                else
                    current_state = 'escapeT';
                end
            end

        case 'FoundT'
            if currentlabel == -1
            elseif (currentlabel == 0 || currentlabel == 5) && segments{current_position,4} > 0.3 || current_position == 1
                current_state = 'UpdatingLabels';
                direction = 1;  % 找到后改变方向向右
                offset = segmentCount ;
            else
                segmentCount = segmentCount + 1;
            end
        case 'UpdatingLabels'
            if currentlabel == -1
            elseif offset > 0
                % 更新标签
                % 条件判断语句
                % currentAmplitude = allAmplitudes(current_position);
                % prevPositions = max(1, current_position - 30):current_position - 1;
                % prevAmplitudes = allAmplitudes(prevPositions);
                % prevLabels = labels(prevPositions);
                % similarLabels = prevLabels(abs(prevAmplitudes - currentAmplitude) < 0.02);
                % if ~isempty(similarLabels)
                %     uniqueLabels= unique(similarLabels);
                %     labelCounts = histcounts(similarLabels,[uniqueLabels - 0.5, max(uniqueLabels) + 0.5]);
                %     [maxCount, maxLabelIdx] = max(labelCounts);
                % else
                %     maxCount = 0;  % 如果similarLabels为空
                % end
                % if maxCount >= 32
                %     LabelToChange = uniqueLabels(maxLabelIdx);
                % else
                    switch segmentCount
                        case 1
                            % 只有一个波，检查是否大于0.5
                            if segments{current_position,6} > 0.5
                                LabelToChange = 5;  % 补动P
                            else
                                LabelToChange = 4;  % 普通P
                            end
                        case 2
                            % 两个波，常见的Q T P Q序列
                            if offset == 1
                                LabelToChange = 4;  % P
                            else
                                if segments{current_position,6}> 0.5
                                    LabelToChange = 2;  % 补动T
                                else
                                    LabelToChange = 1;  % 普通T
                                end
                            end
                        case 3
                            % 三个波，Q T U P Q序列
                            if offset == 1
                                LabelToChange = 4;  % P
                            elseif offset == 2
                                LabelToChange = 3;  % U
                            else
                                LabelToChange = 1;  %普通T
                            end
                        %otherwise
                        %    LabelToChange = -1;
                    end
                % end
                labels(current_position)=LabelToChange;
                if offset == 1
                    current_state = 'escapeQ';  % 完成更新后返回FindT
                end
                offset = offset - 1;
            else
                current_state = 'escapeQ';  % 完成更新后返回FindT
            end
    end
    current_position = current_position + direction;
end
    qrs_peaks = [segments{labels == 0 | labels == 2, 3}]';
% 假设 qrs_peaks 和 ATRTIME 已经存在并且内容如描述所示

% 找到 qrs_peaks 中每个元素在 ATRTIME 中的最近邻
[idx, dist] = knnsearch(ATRTIME, qrs_peaks);

% 设置一个容差，定义匹配的范围（例如正负4）
tolerance = 4;

% 计算查准率和查全率
true_positives = sum(dist <= tolerance); % 符合容差的匹配对数
false_positives = length(qrs_peaks) - true_positives; % qrs_peaks 中没有匹配的元素数
false_negatives = length(ATRTIME) - true_positives; % ATRTIME 中没有匹配的元素数

precision = true_positives / (true_positives + false_positives); % 查准率
recall = true_positives / (true_positives + false_negatives); % 查全率

% 输出结果
fprintf(logFile, 'True Positives: %d\n', true_positives);
fprintf(logFile, 'False Positives: %d\n', false_positives);
fprintf(logFile, 'False Negatives: %d\n', false_negatives);
fprintf(logFile, 'Precision: %.8f\n', precision);
fprintf(logFile, 'Recall: %.8f\n', recall);
%%
    %标签绘制
    h = length(segments)-1; % 曲线段的数量
    %h=5000;
    partplot = h;
    colors{1} = [1,0,0];%           红色
    colors{2} = [0.8,0.8,0.8];%     浅灰色
    colors{3} = [0, 1, 0]; %        绿色
    colors{4} = [0.2, 1, 0.2]; %    蓝色
    colors{5} = [0, 0.75, 0.75];%   青色
    colors{6} = [0.75, 0, 0.75];%   紫色
    colors{7} = [0.6,0.6,0.6];%     深灰色
    colors{8} = [0.75, 0.25, 0.3];% 粉红色

    % 为不同的 ANNOT 类型分配浅色系颜色
    unique_annot = unique(ANNOT);
    annot_colors = containers.Map();
    for i = 1:length(unique_annot)
        annot_colors(unique_annot(i)) = [rand()*0.5+0.5, rand()*0.5+0.5, rand()*0.5+0.5]; % 随机生成浅色系颜色
    end



    for i = 1:h
        label = labels(i); % 获取当前段落的label
        if i >= partplot
            clf; % 清除当前图形窗口的内容
            % 在循环内绘制线和点，并染色
    
            for p = max(1, i-partplot+1):i
                % 提取当前段落的起始和结束位置
                startIdx = segments{p, 1};
                endIdx = segments{p, 2};
                posChange = segments{p, 3} + 5;
                maxvalue = segments{p, 6};
                ptx=segments{p, 7}(:,1);
                pty=segments{p, 7}(:,2);
    
                color = colors{labels(p)+2};
                % 绘制这一段时间的波形和三维相空间图
                % 波形图
                subplot(2, 3, [1, 2, 3]);
                plot(startIdx:endIdx, sigL(startIdx:endIdx), '-', 'Color', color); % 为线段染色
                hold on; % 保持绘图，准备下一个绘图
                plot(posChange, maxvalue, 'o', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerSize', 3); % 为点染色
    
                % 三维相空间图
                subplot(2, 3, 4);
    
                plot(ptx, pty, '-', 'Color', color);
                %plot3(sigL(startIdx:endIdx), sigL(startIdx+tau:endIdx+tau), sigL(startIdx+tau*2:endIdx+tau*2), '-', 'Color', color); % 为线段染色
                hold on; % 保持绘图，准备下一个绘图
                %plot3(sigL(posChange), sigL(posChange+tau), sigL(posChange+2*tau), 'o', 'MarkerEdgeColor', color, 'MarkerFaceColor', color, 'MarkerSize', 3); % 为点染色
    
    
            end
            % 在波形图上标注 ANNOT 和 ATRTIME 信息
            subplot(2, 3, [1, 2, 3]);
            for j = 1:length(ATRTIME)-1
                annot_color = annot_colors(ANNOT(j));
                plot([ATRTIME(j) ATRTIME(j+1)], [1 1],'-', 'Color', annot_color, 'LineWidth', 5);
                plot(ATRTIME(j),0.6 , '+', 'Color', [0, 0, 0.8], 'MarkerSize', 4); % 绘制深蓝色加号
            end

            subplot(2, 3, 5);
            for k = 1:length(sortedIndices)
                labelIdx = sortedIndices(k);
                p = clusterCenterLabel(labelIdx);
                pt = segments{p,7};
                amplitude = segments{p,4};
                range = segments{p,5};
                shrink = min([1, range_A / amplitude, range_D / range]);
                if labels(p) ~= -1
                    color = colors{labels(p)+2};
                    plot(pt(:,1), pt(:,2), '-', 'Color', color);
    
                    hold on;
                    %plot([k*0.01-range*shrink/2,k*0.01+range*shrink/2],[k*0.01+range*shrink/2, k*0.01-range*shrink/2],'color',color, 'LineWidth', 3);
                    plot([k*0.01,k*0.01],[-range*shrink/2, range*shrink/2],'color',color, 'LineWidth', 3);
                end
            end
    
            axis equal;
            grid on;
            axis([-1 1.5 -1 1.5]);
            axis tight;
    
            % 设置波形图和三维相空间图的共通属性
            subplot(2, 3, [1, 2, 3])
            title('波形图');
            xlabel('Time');
            ylabel('Amplitude');
            subplot(2, 3, 4);
    
            title('相空间图');
            xlabel('sigL(i)');
            ylabel(['sigL(i+', num2str(tau),')']);
            %zlabel(['sigL(i+', num2str(2*tau),')']);
            %view([1, 0.5, 1]);
            axis equal;
            grid on;
            axis([-1 1.5 -1 1.5]);
            axis tight;
            
            % 设置图例区域
            subplot(2, 3, 6);
            for j = 1:length(unique_annot)
                annot_color = annot_colors(unique_annot(j));
                % 创建图例用的方块
                rectangle('Position', [0.1, 1.1 - j * 0.1, 0.1, 0.1], 'FaceColor', annot_color, 'EdgeColor', 'none');
                % 在方块内添加文字，居中对齐
                text(0.15, 1.15 - j * 0.1, char(unique_annot(j)), 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 16);
                hold on;
            end
            axis off; % 关闭坐标轴显示

            axis off; % 关闭坐标轴显示

            hold off; % 解除绘图保持状态
            %pause(0.05); % 短暂暂停以便观察
            set(gcf, 'Position', [0, 0, 1920, 1080]); % 位置和大小
            figName = fullfile(dataFolderPath, [num2str(tablemenu_(tablemenu_i)) '.fig']);
            pngName = fullfile(dataFolderPath, [num2str(tablemenu_(tablemenu_i)) '.png']);
            %
            %保存.fig文件
            savefig(figName);
            exportgraphics(gcf, pngName, 'ContentType', 'image');
            %exportgraphics(gcf, pngName, 'Resolution', 96);  % 假定屏幕分辨率为96 DPI
        end
    end
end
% 关闭log文件
fclose(logFile);