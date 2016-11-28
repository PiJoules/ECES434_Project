%% ECES434 Project
% KNN
% Patrick Cross & Leonard Chan

clear;
close all;
clc

[trainMatrix, testMatrix] = partition_data();

%% Calculate KNN
tic

%Take KNN only of every 3rd row since that appears the most valuable. (and
%the last col)
trainMatrixMod = trainMatrix(:,105:156);
testMatrixMod = testMatrix(:,105:156);
trainMatrixMod(:,53)=trainMatrix(:,size(trainMatrix,2));
testMatrixMod(:,53)=testMatrix(:,size(testMatrix,2));  % sets label

%For testing only, reduce size for run time
testObs = 1000;
testMatrixMod = testMatrixMod(1:testObs,:);

%Calculate distance matrix for KNN
features = size(testMatrixMod,2)-1;
d=pdist2(testMatrixMod(:,1:features), trainMatrixMod(:,1:features));

%Create our predictions and calculate accuracies
[sorted,v]=sort(d,2);
for k=1:50
    correct = 0;
    for i=1:testObs
        obs = mode(trainMatrixMod(v(i,1:k),53));
        if obs == testMatrixMod(i,53)
           correct = correct + 1;
        end
    end
    fprintf('Accuracy for KNN size %d = %f, \n',k,correct/testObs);
end
toc