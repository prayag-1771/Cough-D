% The full experimental grid.
%
% Run order: make_val_split.m once, then the four generate_images calls, then
% this. Every run writes one CSV of per recording test decisions into runs/,
% and analyse.py turns that directory into the tables and the paired tests.
%
% Two cells are new. ResNet18 is trained on the linear axis as well as the
% perceptual one, which is what separates the frequency axis from the
% architecture and the initialisation; without those two cells the axis
% effect is not identified by the design. The compact network is trained on
% the perceptual axis for the same reason in reverse.
%
% SEEDS is the part that matters most. With one seed per cell no difference
% below roughly one accuracy point can be interpreted, which is what sank the
% denoising comparison. Five is the minimum worth reporting; more is better.
% Trim SEEDS to [1] for a smoke test before committing to the whole grid.

clc; clear; close all;

SEEDS = [1 2 3 4 5];
OUT   = 'runs';

LINEAR     = 'dataset_linear_images';
LINEAR_W   = 'dataset_wavelet_linear_images';
MEL        = 'dataset_mel_images';
MEL_W      = 'dataset_wavelet_mel_images';

% architecture, image set.  DenseNet201 is the expensive one; it is listed
% last so an interrupted grid still yields the complete cheaper cells.
GRID = {
    'compact',     LINEAR
    'compact',     LINEAR_W
    'compact',     MEL
    'compact',     MEL_W
    'resnet18',    LINEAR
    'resnet18',    LINEAR_W
    'resnet18',    MEL
    'resnet18',    MEL_W
    'mobilenetv2', MEL
    'mobilenetv2', MEL_W
    'densenet201', MEL
    'densenet201', MEL_W
};

total = size(GRID,1) * numel(SEEDS);
done  = 0;
t0    = tic;

for g = 1:size(GRID,1)
    for s = SEEDS

        arch    = GRID{g,1};
        imgbase = GRID{g,2};
        tag     = sprintf('%s__%s__seed%d', arch, imgbase, s);
        target  = fullfile(OUT, [tag '.csv']);

        done = done + 1;

        if exist(target,'file')
            fprintf('[%2d/%2d] skip (exists) %s\n', done, total, tag);
            continue
        end
        if ~exist(fullfile(imgbase,'test'),'dir')
            fprintf('[%2d/%2d] skip (no images) %s\n', done, total, imgbase);
            continue
        end

        fprintf('[%2d/%2d] ', done, total);
        train_one(arch, imgbase, s, OUT);
    end
end

fprintf('\ngrid finished in %.1f minutes\n', toc(t0)/60);
fprintf('now run:  python analyse.py runs\n');
