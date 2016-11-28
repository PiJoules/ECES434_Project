function [trainMatrix, testMatrix]=partition_data()
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

    load('S08.mat')
    allData{8,:} = data;

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
            trainData{i,j}=allData{i,1}{1,j}.X(1:int16(trainRatio*tmpDimX),:);
            testData{i,j}=allData{i,1}{1,j}.X(int16(trainRatio*tmpDimX)+1:end,:);

        end
    end

    trainMatrix = cell2mat(trainData(~cellfun('isempty', trainData))); %Purges empty cells and converts to matrix
    testMatrix = cell2mat(testData(~cellfun('isempty', testData)));
end