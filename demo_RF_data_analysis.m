
clc
clear
close all

disp('Please select the RF file');
[fileRF,pathRF] = uigetfile('*.bin')

% disp('Please select the IQ file');
% [fileIQ,pathIQ] = uigetfile('*.bin');

disp('Please select the ROI file');
[fileROI,pathROI] = uigetfile('*.mat');

% load ROI data
load(fullfile(pathROI,fileROI));

%% reference file
% calculate object depth
centerR = RF_r1 + RF_h/2;
disp(['object depth is row ', num2str(centerR)]);

% select reference file
disp('Please select the reference RF file');
[fileRefer,pathRefer] = uigetfile('*.bin');

% read reference RF data
refer  = readIQRFDataFrame(fullfile(pathRefer,fileRefer),'RF');
refer = cell2mat(refer.data(1,1));
figure,imshow(refer,[]);title('reference signal');

% crop Reference ROI
prompt = "Please input the center of the reference ROI, [center_x,center_y] : ";
temp = input(prompt);
refer = refer( round(temp(2)-RF_h/2):round(temp(2)+RF_h/2), round(temp(1)-RF_w/2):round(temp(1)+RF_w/2) );
figure,imshow(refer,[]);title('croppped reference signal');

%% RF spatial spectrum analysis
% read RF data
RF  = readIQRFDataFrame(fullfile(pathRF,fileRF),'RF');
RF = cell2mat(RF.data(1,1));
figure,imshow(RF,[]);title('sample signal');

% crop ROI
RF = RF( round(RF_r1):round(RF_r1+RF_h), round(RF_c1):round(RF_c1+RF_w) );
figure,imshow(RF,[]);title('cropped sample signal');

% compute spatial spetrum
fs = 50000000; % 采样频率 50MHz
fdezmod = 0; %：解调频率，一般设置为0
Nfft = 512; % fft点数
[spectrum_space,~,f] = rfspect(RF,fs,fdezmod,Nfft);
spectrum_space = spectrum_space( length(spectrum_space)/2 : end );
f = f( length(f)/2 : end );

figure,plot(f,spectrum_space); title('sample spectrum');

[spectrum_refer,~,~] = rfspect(refer,fs,fdezmod,Nfft);
spectrum_refer = spectrum_refer( length(spectrum_refer)/2 : end );
figure,plot(f,spectrum_refer);title('reference spectrum');

% normalization
spectrum_space = spectrum_space./spectrum_refer;
figure,plot(f,spectrum_space); title('corrected spectrum');

% select 1-15MHz
idx = find(f<=15);
f = f(idx);
spectrum_space = spectrum_space(idx);
figure,plot(f,spectrum_space); title('useful spectrum');

% linear fit
[slope_space,intercept_space,midbandfit_space] = spectrum_linear(spectrum_space)

% % S1-S4
% [S1_space,S2_space,S3_space,S4_space] = spectrum_S(spectrum_space);

% HFD
% [nr,nc] = size(RF);
% HFD_sum = 0;
% for ci = 1:nc
%     line_cur = squeeze(RF(:,ci));
%     HFD_sum = HFD_sum + Higuchi_FD(line_cur, 10);
% end
% HFD_space = HFD_sum/(nc);


%% RF temporal spectrum analysis
fileRF = dir([pathRF,'\*.bin']);
RF_series = [];
for fi = 1:length(fileRF)
    fname = dir([pathRF,'\B_RF_Frame_',num2str(fi),'_*.bin']);
    RF  = readIQRFDataFrame(fullfile(fname.folder,fname.name),'RF');
    RF = cell2mat(RF.data(1,1));
    RF = RF( round(RF_r1):round(RF_r1+RF_h), round(RF_c1):round(RF_c1+RF_w) );
    RF_series = cat(3,RF_series,RF);
end

% 计算RF series中每个像素的频谱 （双边）
fs = 20; % 采样频率 一秒钟采集了多少张RF图像
fdezmod = 0; %：解调频率，一般设置为0
Nfft = 512; % fft点数
[spectrum_time,~,~]=rfspect_time(RF_series,fs,fdezmod,Nfft);
% figure,plot(f,spectrum_time);xlabel('f/Hz');ylabel('average spectrum /db');
spectrum_time = spectrum_time( length(spectrum_time)/2 : end );

% 参数S1-S4
[S1_time,S2_time,S3_time,S4_time] = spectrum_S(spectrum_time);

% 线性拟合
[slope_time,intercept_time,midbandfit_time] = spectrum_linear(spectrum_time);

% HFD
[nr,nc,nz] = size(RF_series);
n_pixels = nr*nc;
HFD_sum = 0;
for ri = 1:nr
    for ci = 1:nc
        RF_cur = squeeze(RF_series(ri,ci,:));
        HFD_sum = HFD_sum + Higuchi_FD(RF_cur, 10);
    end
end
HFD_time = HFD_sum/n_pixels;

%% RF envelope statistics
IQ  = readIQRFDataFrame(fullfile(pathIQ,fileIQ),'IQ');
IQ = cell2mat(IQ.data(1,1));

% crop ROI
IQ = IQ( round(IQ_r1):round(IQ_r1+IQ_h), round(IQ_c1):round(IQ_c1+IQ_w) );
env = abs(IQ);
env = log(env);
% Nakagami distribution
env = double(env);
env = env(env>0);
env = env./max(env);
figure, histogram(env,500,'Normalization','pdf');
pd = fitdist(env,'Nakagami');
Nakagami_mu = pd.mu;
Nakagami_omega = pd.omega;


%% create final parametric table
params = table(Nakagami_mu,Nakagami_omega,S1_space,S2_space,S3_space,S4_space,...
    slope_space,intercept_space,midbandfit_space,HFD_space,...
    S1_time,S2_time,S3_time,S4_time,slope_time,intercept_time,midbandfit_time,HFD_time);



