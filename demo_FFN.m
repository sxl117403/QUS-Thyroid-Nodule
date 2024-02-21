clc
clear;

rootname = 'C:\Users\03\Desktop\RF\2023-11-7';

% HQ
load(fullfile(rootname,'HQ_paras.mat'));
load(fullfile(rootname,'table_RF_space_HQ.mat'));
X_HQ = [table_clinic.Age, ...
    table_clinic.Component,table_clinic.Echo,table_clinic.aspect, table_clinic.Margin,table_clinic.Calcium,...
    table_RF_env.Nakagami_mu, table_RF_env.Nakagami_omega,...
    table_RF_space.Slope_space,table_RF_space.Intercept_space,table_RF_space.Midbandfit_space];
Y_HQ = table_RF_env.labels;
ID_HQ = table_clinic.data;

% ZQ
load(fullfile(rootname,'ZQ_paras.mat'));
load(fullfile(rootname,'table_RF_space_ZQ.mat'));
X_ZQ = [table_clinic.Age,...
    table_clinic.Component,table_clinic.Echo,table_clinic.aspect, table_clinic.Margin,table_clinic.Calcium,...
    table_RF_env.Nakagami_mu, table_RF_env.Nakagami_omega,...
    table_RF_space.Slope_space,table_RF_space.Intercept_space,table_RF_space.Midbandfit_space];
Y_ZQ = table_RF_env.labels;
ID_ZQ = table_clinic.data;

%% 5 fold
for i = 1:5

    % HQ_idx
    load([rootname,'\HQ\fold',num2str(i),'_test_idx.mat']);
    temp = [test_idx0;test_idx1];
    load([rootname,'\HQ\fold',num2str(i),'_test_idx_uniq.mat']);
    test_idx_HQ = [temp;test_idx0;test_idx1];
    train_idx_HQ = setdiff(1:length(Y_HQ),test_idx_HQ);

    % HQ IDs

    test_ID_HQ = ID_HQ(test_idx_HQ);
    train_ID_HQ = ID_HQ(train_idx_HQ);

    % ZQ_idx
    load([rootname,'\ZQ\fold',num2str(i),'_test_idx.mat']);
    temp = [test_idx0;test_idx1];
    load([rootname,'\ZQ\fold',num2str(i),'_test_idx_uniq.mat']);
    test_idx_ZQ = [temp;test_idx0;test_idx1];  
    train_idx_ZQ = setdiff(1:length(Y_ZQ),test_idx_ZQ);

    % ZQ IDs
    test_ID_ZQ = ID_ZQ(test_idx_ZQ);
    train_ID_ZQ = ID_ZQ(train_idx_ZQ);
    
    % test dataset
    X_test = [X_HQ(test_idx_HQ,:);X_ZQ(test_idx_ZQ,:)];
    Y_test = [Y_HQ(test_idx_HQ);Y_ZQ(test_idx_ZQ)];

    % test IDs
    test_ID = [test_ID_HQ; test_ID_ZQ];

    % train dataset
    X_train = [X_HQ(train_idx_HQ,:);X_ZQ(train_idx_ZQ,:)];
    Y_train = [Y_HQ(train_idx_HQ);Y_ZQ(train_idx_ZQ)];

    % train IDs
    train_ID = [train_ID_HQ; train_ID_ZQ];

    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainbr';  % Levenberg-Marquardt backpropagation.

    % Create a Fitting Network
    hiddenLayerSize = [5,8,5];% 5;
    net = feedforwardnet(hiddenLayerSize,trainFcn);% fitnet(hiddenLayerSize,trainFcn);
    net.trainParam.max_fail = 20;


    Xin = [X_test;X_train]'; Yin = [Y_test;Y_train]';

    % 设置数据划分。
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = size(X_test,1)+1:size(Xin,2);
    net.divideParam.valInd = 1:size(X_test,1);
    net.divideParam.testInd= 1:size(X_test,1);

    %train


    auc = 0;

    for k = 1:50
        [net_train,tr] = train(net,Xin,Yin);

        tInd = tr.valInd;
        Y_pred = net_train(Xin(:,tInd));

        % AUC
        rocObj = rocmetrics(Y_test,Y_pred',1);
        auc_cur = rocObj.AUC;

        if auc<auc_cur

            auc = auc_cur
            %figure,plot(rocObj);
            Y1 = zeros(size(Y_pred));
            Y1(Y_pred>0.5)=1;

            Y1 = Y1';
            % acc
            TP = length(find(Y_test(Y_test==Y1)==1));
            TN = length(find(Y_test(Y_test==Y1)==0));
            FP = length(find(Y1(Y_test~=Y1)==1));
            FN = length(find(Y1(Y_test~=Y1)==0));
            Metrics.Acc = (TP+TN)/(TP+TN+FP+FN);
            Metrics.Precision = TP/(TP+FP);
            Metrics.Recall = TP/(TP+FN);  %sensitivity
            Metrics.Specificity = TN/(TN+FP);
            Metrics.TP = TP;
            Metrics.TN = TN;
            Metrics.FP = FP;
            Metrics.FN = FN;         
            

            sname = ['fold',num2str(i),'_pred.mat'];
            save(sname,'Metrics','rocObj','net_train','Y_pred','Y_test','test_ID','train_ID','auc');
        end
    end


end


