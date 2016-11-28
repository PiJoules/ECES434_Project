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

for k=1:10
    mdl = fitcknn(trainMatrixMod(:,1:52), trainMatrixMod(:,53), 'NumNeighbors', k);
    label = predict(mdl, testMatrixMod(:,1:52));

    correct = 0;
    testObjs = size(testMatrixMod, 1);
    for i=1:testObs
        if label(i) == testMatrixMod(i,53)
            correct = correct + 1;
        end
    end
    fprintf('Accuracy for KNN at k = %d: %f, \n', k,correct/testObs);
end
toc