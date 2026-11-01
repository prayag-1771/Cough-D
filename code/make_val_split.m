% Carve a validation partition out of the existing training partition.
%
% The test partition is left exactly as it is, so every number produced after
% this runs stays directly comparable with the numbers already reported. Only
% the training material is divided, which is what was missing: the test set
% was previously passed to trainingOptions as ValidationData, so the material
% used to watch training was also the material used to report accuracy.
%
% Run this ONCE. It moves files. The seed is fixed so the division is
% reproducible, and split_manifest.csv records where every file went.

clc; clear; close all;

rng(0);                         % fixed: the division must be reproducible

src     = fullfile('dataset','train');
dst     = fullfile('dataset','val');
classes = {'cough','non_cough'};
val_fraction = 0.15;

assert(exist(src,'dir')==7, 'dataset/train not found: run this from the project root');
assert(exist(dst,'dir')~=7, 'dataset/val already exists: refusing to split twice');

rows = {};

for c = 1:numel(classes)

    in_folder  = fullfile(src, classes{c});
    out_folder = fullfile(dst, classes{c});

    if ~exist(out_folder,'dir')
        mkdir(out_folder);
    end

    files = dir(fullfile(in_folder,'*.wav'));
    n     = numel(files);
    k     = round(val_fraction * n);

    % randperm over a fixed seed: the same k files move on every machine
    moved = randperm(n, k);
    keep  = setdiff(1:n, moved);

    for i = moved
        movefile(fullfile(in_folder, files(i).name), ...
                 fullfile(out_folder, files(i).name));
        rows(end+1,:) = {files(i).name, classes{c}, 'val'};     %#ok<SAGROW>
    end

    for i = keep
        rows(end+1,:) = {files(i).name, classes{c}, 'train'};   %#ok<SAGROW>
    end

    fprintf('%-10s  train %5d   val %5d\n', classes{c}, n-k, k);
end

% the test partition is untouched, but record it so the manifest is complete
for c = 1:numel(classes)
    files = dir(fullfile('dataset','test',classes{c},'*.wav'));
    for i = 1:numel(files)
        rows(end+1,:) = {files(i).name, classes{c}, 'test'};    %#ok<SAGROW>
    end
end

T = cell2table(rows, 'VariableNames', {'file','class','partition'});
writetable(T, 'split_manifest.csv');

fprintf('\nwrote split_manifest.csv  (%d recordings)\n', height(T));
fprintf('the test partition was not modified\n');
