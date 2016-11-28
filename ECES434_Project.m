%% ECES434 Project
% Mental Math
% Patrick Cross & Leonard Chan

clear all;
close all;
clc

%% Load data for processing
load('S01.mat')
allData{1,:} = data;

load('S02.mat')
allData{2,:} = data;

load('S03.mat')
allData{3,:} = data;

load('S04.mat')
allData{4,:} = data;

load('S05.mat')
allData{5,:} = data;

load('S06.mat')
allData{6,:} = data;

load('S07.mat')
allData{7,:} = data;

clear data

%% Split the data into training and testing set
% Since trials have semi random start and end times I'm going to grab
% random subset of trials
%Note to self need to clear unused cells/matricies when done with them
numPatients = size(allData,1);

indx = randperm(numPatients);
allData=allData(indx);

trainRatio = 0.8;   %Ratio of data to be train / test

answerCol = size(allData{1,1}{1,1}.X,2)+1; %Col position of sub or rest

testData={1};   %Initialize train and test data as cells
trainData={1};

for i=1:numPatients     %Loop through all patients in data set
    for j=1:size(allData{i},2)  %Loop through num of 6 trials for patient
        %Add labels for subtracting or resting (1 = sub, -1 = rest)
        for k=1:size(allData{i,1}{1,j}.X,1)
            
            %Most hidious if ever but matlab can't range matricies
            if ((k>=allData{i,1}{1,j}.trial(1) && k<allData{i,1}{1,j}.trial(2)) ||...
                    (k>=allData{i,1}{1,j}.trial(3) && k<allData{i,1}{1,j}.trial(4))||...
                    (k>=allData{i,1}{1,j}.trial(5) && k<allData{i,1}{1,j}.trial(6))||...
                    (k>=allData{i,1}{1,j}.trial(7) && k<allData{i,1}{1,j}.trial(8))||...
                    (k>=allData{i,1}{1,j}.trial(9) && k<allData{i,1}{1,j}.trial(10))||...
                    (k>=allData{i,1}{1,j}.trial(11) && k<allData{i,1}{1,j}.trial(12)))
                
                allData{i,1}{1,j}.X(k,answerCol)=1;  %Mental math
            else
                allData{i,1}{1,j}.X(k,answerCol)=-1; %No math
            end
        end
        
        %Shuffel the matrix
        tmpDimX = size(allData{i,1}{1,j}.X,1);
        indx = randperm(tmpDimX);
        allData{i,1}{1,j}.X = allData{i,1}{1,j}.X(indx,:);
        
        %Split into training and test Cells (makes easier to process)
        trainData{i,j}=allData{i,1}{1,j}.X(1:int16(0.8*tmpDimX),:);
        testData{i,j}=allData{i,1}{1,j}.X(int16(0.8*tmpDimX)+1:end,:);

    end
end

trainMatrix = cell2mat(trainData(~cellfun('isempty', trainData))); %Purges empty cells and converts to matrix
testMatrix = cell2mat(testData(~cellfun('isempty', testData)));

%Clear unused data sets that were temporary
clear trainData;
clear testData;
clear allData;


%% Calculate KNN
%DO NOT RUN ON FULL DATA SET. WAY TO MUCH RAM NEEDED
% This is raw KNN it takes WAY TO LONG to calculate
% d=zeros(size(testMatrix,1),size(trainMatrix,1));
% for i=1:size(testMatrix,1)
%     for j=1:size(trainMatrix,1)
%         d(i,j)=sqrt(sum((testMatrix(i,1:size(testMatrix,2)-1)-trainMatrix(j,1:size(trainMatrix,2)-1)).^2));
%     end
% end

%Take KNN only of every 3rd row since that appears the most valuable. (and
%the last col)
%trainMatrixMod = trainMatrix(:,3:3:size(trainMatrix,2));
%testMatrixMod = testMatrix(:,3:3:size(testMatrix,2));
trainMatrixMod = trainMatrix(:,105:156);
testMatrixMod = testMatrix(:,105:156);
trainMatrixMod(:,53)=trainMatrix(:,size(trainMatrix,2));
testMatrixMod(:,53)=testMatrix(:,size(testMatrix,2));

%For testing only, Reduce size for run time
testMatrixMod = testMatrixMod(1:1000,:);

%Calculate distance matrix for KNN
d=zeros(size(testMatrixMod,1),size(trainMatrixMod,1));
for i=1:size(testMatrixMod,1)
    % disp(i) %Used to display i to show time.
    for j=1:size(trainMatrixMod,1)
        d(i,j)=sqrt(sum((testMatrixMod(i,1:size(testMatrixMod,2)-1)-trainMatrixMod(j,1:size(trainMatrixMod,2)-1)).^2));
    end
end

%Create our predictions and calculate accuracies
[sorted,v]=sort(d,2);
predict1=zeros(size(testMatrixMod,1),1);
for j=1:1:20
    for i=1:size(testMatrixMod,1)
        predict1(i)=mode(trainMatrixMod(v(i,1:j),53));  %Make prediction with varrying number of neighbors
    end
	[FP, FN, TP, TN, acc, prec, rec, f_meas, TPR, FPR] = performance(predict1,testMatrixMod(:,53),0);
    fprintf('Accuracy for KNN size %f = %f, \n',j,acc);
end

%Plot the index in the test data set where the 10 shortest distances
%occurreed.
plot(1:10,v(:,1:10))
xlabel('Distance index')
ylabel('Index in train set')
title('Display showing distribution of used train set for test set')

figure;
plot(1:10,v(:,1:10))
xlabel('Distance index')
ylabel('Index in train set')
title('Display showing distribution of used train set for test set')
axis([1 10 1 5000]);

disp 'completed KNN';