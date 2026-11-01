function train_one(arch, imgbase, seed, outdir)
% Train one architecture on one image set with one seed, and save the per
% recording test decisions.
%
%   train_one('resnet18', 'dataset_wavelet_mel_images', 1, 'runs')
%
% Three changes from the original training scripts, each addressing something
% a reviewer will otherwise raise.
%
%   1. ValidationData is the validation partition, not the test partition.
%      Previously the test set was passed as ValidationData, which meant the
%      material used to monitor training was the material used to report
%      accuracy. make_val_split.m creates the partition this needs.
%
%   2. rng(seed) is set, and seed is a parameter. Running the grid over
%      several seeds gives the spread across initialisations, without which
%      no difference smaller than about one accuracy point can be
%      interpreted. This is the single most important addition.
%
%   3. The per recording decisions are written to CSV. McNemar's test needs
%      the two discordant counts, which are properties of the paired
%      decisions and cannot be recovered from a confusion matrix. Saving
%      them is what makes paired significance testing possible at all.
%
% The network retained is the one with the best validation score rather than
% the one at the final iteration.

narginchk(4,4);

if ~exist(outdir,'dir')
    mkdir(outdir);
end

rng(seed);                      % governs initialisation and shuffling

imdsTrain = imageDatastore(fullfile(imgbase,'train'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsVal   = imageDatastore(fullfile(imgbase,'val'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest  = imageDatastore(fullfile(imgbase,'test'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

numClasses = numel(categories(imdsTrain.Labels));
assert(numClasses==2, 'expected two classes, found %d', numClasses);

% ---------------------------------------------------------------- the model
if strcmpi(arch,'compact')

    inputSize = [224 224 3];
    lgraph = [
        imageInputLayer(inputSize)
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(128)
        reluLayer
        dropoutLayer(0.5)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    lr = 1e-3;
    epochs = 10;

else
    switch lower(arch)
        case 'resnet18'
            net = resnet18;
            fcName = 'fc1000'; smName = 'prob';
            clName = 'ClassificationLayer_predictions';
        case 'mobilenetv2'
            net = mobilenetv2;
            fcName = 'Logits'; smName = 'Logits_softmax';
            clName = 'ClassificationLayer_Logits';
        case 'densenet201'
            net = densenet201;
            fcName = 'fc1000'; smName = 'fc1000_softmax';
            clName = 'ClassificationLayer_fc1000';
        otherwise
            error('unknown architecture: %s', arch);
    end

    inputSize = net.Layers(1).InputSize;
    lgraph = layerGraph(net);
    lgraph = replaceLayer(lgraph, fcName, ...
        fullyConnectedLayer(numClasses,'Name','fc_cough', ...
            'WeightLearnRateFactor',10,'BiasLearnRateFactor',10));
    lgraph = replaceLayer(lgraph, smName, softmaxLayer('Name','softmax'));
    lgraph = replaceLayer(lgraph, clName, classificationLayer('Name','classoutput'));

    % every transferred model now gets the same budget, so the comparison
    % among them is no longer confounded with the schedule
    lr = 1e-3;
    epochs = 10;
end

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing','gray2rgb');
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal,   'ColorPreprocessing','gray2rgb');
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest,  'ColorPreprocessing','gray2rgb');

options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',lr, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',50, ...
    'OutputNetwork','best-validation', ...
    'ExecutionEnvironment','gpu', ...
    'Plots','none', ...
    'Verbose',false);

trained = trainNetwork(augTrain, lgraph, options);

% ------------------------------------------------- the test decisions, saved
[YPred, scores] = classify(trained, augTest);
YTrue = imdsTest.Labels;

[~, name, ext] = cellfun(@fileparts, imdsTest.Files, 'UniformOutput', false);
fname = strcat(name, ext);

T = table(fname, string(YTrue), string(YPred), scores(:,1), scores(:,2), ...
    'VariableNames', {'file','true','pred','score_1','score_2'});

tag = sprintf('%s__%s__seed%d', lower(arch), imgbase, seed);
writetable(T, fullfile(outdir, [tag '.csv']));

fprintf('%-46s acc %.4f   -> %s.csv\n', tag, mean(YPred==YTrue), tag);
end
