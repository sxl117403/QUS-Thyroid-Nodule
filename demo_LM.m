
% 利用MATLAB对数据进行逻辑回归分析，分别采用逐步寻优（逐步剔除掉最不显著的因变量）和使用MATLAB自带的逐步向前、向后回归函数进行建模。
% https://www.cnblogs.com/cruelty_angel/p/11452726.html

%% 逻辑回归 自动建模
clc
clear;

rootname = 'C:\Users\03\Desktop\RF\2023-11-7';

% HQ
load(fullfile(rootname,'HQ_paras.mat'));
load(fullfile(rootname,'table_RF_space_HQ.mat'));
X_HQ = [table_clinic.Age, table_clinic.Component,table_clinic.Echo,table_clinic.aspect,...
    table_clinic.Margin,table_clinic.Calcium];
Y_HQ = table_RF_env.labels;
ID_HQ = table_clinic.data;

% ZQ
load(fullfile(rootname,'ZQ_paras.mat'));
load(fullfile(rootname,'table_RF_space_ZQ.mat'));
X_ZQ = [table_clinic.Age, table_clinic.Component,table_clinic.Echo,table_clinic.aspect,...
    table_clinic.Margin,table_clinic.Calcium];
Y_ZQ = table_RF_env.labels;
ID_ZQ = table_clinic.data;

[ID_same,idx_same_HQ,idx_same_ZQ] = intersect(ID_HQ,ID_ZQ);

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
    load([rootname,'\ZQ\fold',num2str(i),'_test_idx_uniq.mat']);
    test_idx_ZQ = [test_idx0;test_idx1];
    train_idx_ZQ = setdiff(1:length(Y_ZQ),[test_idx_ZQ;idx_same_ZQ]);

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


    %train
    mdl = fitglm(X_train,Y_train,'linear','Distribution','binomial','Link','logit');
    Y_pred = predict(mdl,X_test);   
   
    Y1 = zeros(size(Y_pred));
    Y1(Y_pred>0.5)=1;

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

    % AUC
    rocObj = rocmetrics(Y_test,Y_pred,1);


    sname = ['fold',num2str(i),'_pred.mat'];
    save(sname,'Metrics','rocObj','mdl','Y_pred','Y_test','test_ID','train_ID','rocObj');




end


