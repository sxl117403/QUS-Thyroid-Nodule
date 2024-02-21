clc
clear
close all
rPath = 'D:\Projects\RF\dataset\GE\2_ROI_data\IQ\';

labels = dir([rPath,'\*']);
labels = labels(3:end);

n = 0;


for label_i = 1:2
    label_name = labels(label_i).name;
    subjects = dir([rPath,label_name,'\*']);
    subjects = subjects(3:end);

    for si = 1:length(subjects)
        sname = subjects(si).name;
        datasets = dir([rPath,label_name,'\',sname,'\*']);
        datasets = datasets(3:end);
        for di = 1:length(datasets)


            dname = datasets(di).name;

            files = dir([rPath,label_name,'\',sname,'\',dname,'\*.mat']);
            fname = fullfile(rPath,label_name,sname,dname,files(1).name);

            load(fname);
            env = abs(IQ_cur_ROI);
            env = log(env);
            % Nakagami distribution
            env = double(env);
            env = env(env>0);
           % env = env./max(env);
           
            pd = fitdist(env,'Nakagami');
            Nakagami_mu = pd.mu;
            Nakagami_omega = pd.omega;


            x_values = min(env(:)):0.01:max(env(:));
            y = pdf(pd,x_values);
            figure,histogram(env,'Normalization','pdf');
            hold on, plot(x_values,y);


            n = n+1;
            ID{n,1} = sname;
            label(n,1) = str2num(label_name);
            dataset{n,1} = dname;
            Mu(n,1) = Nakagami_mu;
            Omega(n,1) = Nakagami_omega;
            n

        end
    end


end

results = table(ID,label,dataset,Mu,Omega);
save('results.mat','results');


% ttest
group0_mu = Mu(label==0); 
group1_mu = Mu(label==1);
mu0_mean = mean(group0_mu)
mu0_std = std(group0_mu)
mu1_mean = mean(group1_mu)
mu1_std = std(group1_mu)
[stats,p] = ttest2(group0_mu,group1_mu);
mu_p = p

group0_omega = Omega(label==0);
group1_omega = Omega(label==1);

omega0_mean = mean(group0_omega)
omega0_std = std(group0_omega)

omega1_mean = mean(group1_omega)
omega1_std = std(group1_omega)
[stats,p] = ttest2(group0_omega,group1_omega);
omega_p = p

