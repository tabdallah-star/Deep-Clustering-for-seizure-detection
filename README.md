# Deep-Clustering-for-seizure-detection
1) Read the data:
   
   The publi UCI Epileptic Seizure Dataset can be found online: https://www.kaggle.com/code/maximkumundzhiev/epileptic-seizure-recognition
   The CHU dataset is a local private dataset

2) Slidding Window with 30% overlap to the CHU dataset:
   
   INPUT: The raw EEG data
   OUTPUT: The series of overlapping data segments
   clear all
clc

load('event.mat')
load('200312W-AEX_0005.mat')
fileID = fopen('200312W-AEX_0005Info.txt','r');
nbevt = fscanf(fileID,'[nbevt] %d\n\n',1);
EEGSignal=EEG([1:17], :); %We will work only on the 18 commun channels 
windowidth=1;% slidding windows width in sec
Fs=256; % sampling Frequency
ns= Fs*windowidth; % number of samples in each windows
OL=0.7; % percentage of overlap in case of non-seizure
OLS=0.7; % percentage of overlap in case of seizure
no=ceil(OL*ns); % number of overlapping samples in case of non-seizure
nos=ceil(OLS*ns); % number of overlapping samples in case of seizure
j=1;
for p=1:nbevt
if p==1 starti=1;
else starti=eventdata(p-1).tEnd*Fs+1;
end
for i=starti:no:eventdata(p).t0*Fs-ns 
Signal{j}= EEGSignal(:,i:i+ns-1);
label{j}=0;
startindex{j}=i; %in sample not seconds
Endindex{j}=i+ns-1;

j=j+1;
end
for i=(eventdata(p).t0+1)*Fs:nos:eventdata(p).tEnd*Fs-ns    
Signal{j}= EEGSignal(:,i:i+ns-1);
label{j}=1;
startindex{j}=i; %in sample not seconds
Endindex{j}=i+ns-1; 
j=j+1;
end
if p==nbevt 
for i=(eventdata(p).tEnd-1)*Fs+1:no:length(EEGSignal)-ns 
Signal{j}= EEGSignal(:,i:i+ns-1);
label{j}=0;
startindex{j}=i; %in sample not seconds
Endindex{j}=i+ns-1;
j=j+1;
end
j=j-1;
end
end
count = 0;

% Iterate through each cell
for i = 1:numel(label)
    % Check if the cell contains a 1
    if isequal(label{i}, 1)
        % Increment counter
        count = count + 1;
    end
end

% Display the total count
disp(['Total count of 1: ', num2str(count)]);
