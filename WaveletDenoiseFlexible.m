function DenoisingSignal = WaveletDenoiseFlexible(ecgdata, n, wavename, zeroDetailLevels)
    % 进行小波分解
    [C,L] = wavedec(ecgdata, n, wavename);

    % 初始化细节系数为零的操作
    for i = 1:zeroDetailLevels
        eval(sprintf('D%d = zeros(size(detcoef(C,L,i)));', i));
    end
    
    % 获取最后一级的近似系数并置零（去除基线漂移）
    A = appcoef(C,L,wavename,n);
    A = zeros(size(A));
    
    % 从最后一级开始重构信号
    for i = n:-1:1
        if i > zeroDetailLevels
            D = detcoef(C,L,i);
        else
            eval(sprintf('D = D%d;', i));
        end
        
        if i == n
            % 对于第一次迭代使用置零的最后一级近似系数
            RA = idwt(A, D, wavename);
        else
            % 对于其他迭代使用上一步的重构结果
            RA = idwt(RA(1:length(D)), D, wavename);
        end
    end
    
    % 最终重构的信号
    DenoisingSignal = RA;
end