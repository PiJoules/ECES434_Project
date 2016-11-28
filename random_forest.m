%% ECES434 Project
% Random Forest
% Patrick Cross & Leonard Chan

clear;
close all;
clc;

rng(1);  % Reproducability

[trainMatrix, testMatrix] = partition_data();

%% Calculate KNN

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
trainObs = size(trainMatrixMod,1);
%trainObs = 1000;  % Testing with subset of training shows decrease in accuracy

for k=1:10
    Mdl = TreeBagger(k,trainMatrixMod(1:trainObs,1:features),trainMatrixMod(1:trainObs,features+1),'OOBPrediction','On','Method','classification');

    predictions = predict(Mdl, testMatrixMod(1:testObs, 1:features));
    correct = 0;
    for i=1:testObs
        if str2num(predictions{i}) == testMatrixMod(i,features+1)
            correct = correct + 1;
        end
    end
    fprintf('Accuracy for Random Forest with %d tree(s) = %f, \n',k,correct/testObs);
end
