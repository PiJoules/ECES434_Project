%% ECES434 Project
% KNN
% Patrick Cross & Leonard Chan

% Note for the sake of processing time you can comment out
% any section that you do not want to run

clear;
close all;
clc


%% Calculations for KNN
[trainMatrix, testMatrix] = partition_data();

%Quick justification of data seperation
mean(var(trainMatrix(:,1:52)))
mean(var(trainMatrix(:,53:104)))
mean(var(trainMatrix(:,105:156)))

tic

%Take KNN only of every 3rd row since that appears the most valuable. (and
%the last col)
trainMatrixMod = trainMatrix(:,105:156);
testMatrixMod = testMatrix(:,105:156);
trainMatrixMod(:,53)=trainMatrix(:,size(trainMatrix,2));
testMatrixMod(:,53)=testMatrix(:,size(testMatrix,2));  % sets label

%For testing only, reduce size for run time
testObs = 2000;
testMatrixMod = testMatrixMod(1:testObs,:);

%Calculate distance matrix for KNN
features = size(testMatrixMod,2)-1;
d=pdist2(testMatrixMod(:,1:features), trainMatrixMod(:,1:features));

%Create our predictions and calculate accuracies
[sorted,v]=sort(d,2);
accTable = zeros(50,1);
for k=1:50
    correct = 0;
    for i=1:testObs
        obs = mode(trainMatrixMod(v(i,1:k),53));
        if obs == testMatrixMod(i,53)
           correct = correct + 1;
        end
    end
    accTable(k)=correct/testObs;
    fprintf('Accuracy for KNN size %d = %f, \n',k,correct/testObs);
end
toc
figure;
plot(accTable);
xlabel('num K neighbors');
ylabel('Accuracy');

%% Calculate for Random Forest
clear all;
[trainMatrix, testMatrix] = partition_data();

trainMatrixMod = trainMatrix(:,105:156);
testMatrixMod = testMatrix(:,105:156);
trainMatrixMod(:,53)=trainMatrix(:,size(trainMatrix,2));
testMatrixMod(:,53)=testMatrix(:,size(testMatrix,2));  % sets label

%For testing only, reduce size for run time
testObs = 1000;
testMatrixMod = testMatrixMod(1:testObs,:);

features = size(testMatrixMod,2)-1;
trainObs = size(trainMatrixMod,1);
trainObs = 1000;  % Testing with subset of training shows decrease in accuracy

accTable = zeros(10,1);
tic
for k=1:10
    Mdl = TreeBagger(k,trainMatrixMod(1:trainObs,1:features),trainMatrixMod(1:trainObs,features+1),'OOBPrediction','On','Method','classification');

    predictions = predict(Mdl, testMatrixMod(1:testObs, 1:features));
    correct = 0;
    for i=1:testObs
        if str2num(predictions{i}) == testMatrixMod(i,features+1)
            correct = correct + 1;
        end
    end
    accTable(k)=correct/testObs;
    fprintf('Accuracy for Random Forest with %d tree(s) = %f, \n',k,correct/testObs);
end
toc
figure;
plot(accTable);
xlabel('num of trees');
ylabel('Accuracy');

%% Calculate for MLP
clear all;

[trainMatrix, testMatrix] = partition_data();

trainMatrixMod = trainMatrix(:,105:156);
testMatrixMod = testMatrix(:,105:156);
trainMatrixMod(:,53)=trainMatrix(:,size(trainMatrix,2));
testMatrixMod(:,53)=testMatrix(:,size(testMatrix,2));  

%testObs = 10000;
%testMatrixMod = testMatrixMod(1:testObs,:);

features = size(testMatrixMod,2)-1;

numLayersMax = 10;
threshPercision = 10;
accTable = zeros(numLayersMax,threshPercision);
timeTable = zeros(numLayersMax,1);
for i = 1:numLayersMax
    tic
    net = newff(trainMatrixMod(:,1:52)',trainMatrixMod(:,53)',i); % hidden nodes = 10
    net = init(net); % Neural network initialization
    net = train(net,trainMatrixMod(:,1:52)',trainMatrixMod(:,53)'); % train network
    % example of classifying test samples by ANN:
    predicted = sim(net,testMatrixMod(:,1:52)');
    T=toc;
    timeTable(i)=T;
    % evaluate the performance
    f = predicted;
    f = f-min(f(:));
    f = f ./ max(f(:));
    for j = 0:1/threshPercision:1
        [FP, FN, TP, TN, acc, prec, rec, f_meas, TPR, FPR] = performance(f,testMatrixMod(:,53),j);
        accTable(i,int16(j*threshPercision+1))=acc;
    end
end
figure;
surf([0:0.1:1],[1:10],accTable);
xlabel('threshold')
ylabel('num hidden layers')
zlabel('accuracy')

figure;
plot(timeTable);
xlabel('num hidden layers')
ylabel('Time to create (s)')