function generate_images(axis_kind, denoise)
% Build spectrogram images for one experimental condition.
%
%   generate_images('linear', false)      -> dataset_linear_images
%   generate_images('linear', true)       -> dataset_wavelet_linear_images
%   generate_images('mel',    false)      -> dataset_mel_images
%   generate_images('mel',    true)       -> dataset_wavelet_mel_images
%
% Two things differ from generate_stft_images.m, and both matter.
%
% First, the linear axis transform is now taken from the return value of
% spectrogram rather than by plotting it and calling getframe. The previous
% version handed the network a raster capture of a figure window, which
% carried the display colour map and whatever resolution the rendering
% surface happened to have, and which cannot be regenerated on another
% machine. Capturing the matrix instead makes the input the transform itself
% and makes it reproducible.
%
% Second, the linear axis branch now resamples to 16 kHz exactly as the mel
% branch does. Previously it analysed each recording at whatever rate it was
% recorded at, so the two branches were not comparable.
%
% Everything after the transform is identical across the two axes: log
% compression, mapping to the unit interval, resize to 224 by 224, written as
% a single channel PNG. The two conditions therefore differ in the frequency
% axis and in nothing else, which is what isolating the axis effect requires.

narginchk(2,2);
validateattributes(denoise, {'logical'}, {'scalar'});

Fs_target = 16000;
classes   = {'cough','non_cough'};
sets      = {'train','val','test'};
window    = hamming(512);
overlap   = 256;
fftlen    = 512;

switch lower(axis_kind)
    case 'linear', base = 'dataset_linear_images';
    case 'mel',    base = 'dataset_mel_images';
    otherwise, error('axis_kind must be ''linear'' or ''mel''');
end
if denoise
    base = strrep(base, 'dataset_', 'dataset_wavelet_');
end

fprintf('axis=%s  denoise=%d  ->  %s\n', axis_kind, denoise, base);

if isempty(gcp('nocreate'))
    parpool;
end

for s = 1:numel(sets)
    for c = 1:numel(classes)

        in_folder  = fullfile('dataset', sets{s}, classes{c});
        out_folder = fullfile(base,      sets{s}, classes{c});

        if ~exist(in_folder,'dir')
            warning('missing %s, skipped', in_folder);
            continue
        end
        if ~exist(out_folder,'dir')
            mkdir(out_folder);
        end

        files = dir(fullfile(in_folder,'*.wav'));

        parfor i = 1:numel(files)

            [x, Fs] = audioread(fullfile(in_folder, files(i).name));

            if size(x,2) > 1
                x = mean(x,2);
            end
            if Fs ~= Fs_target
                x  = resample(x, Fs_target, Fs);
                Fs = Fs_target;
            end

            if denoise
                % sym4, level 6, empirical Bayes with a Cauchy prior and a
                % posterior median rule. These are wdenoise's defaults, and
                % they are named here so the manuscript can state them.
                x = wdenoise(x, 6, ...
                        'Wavelet','sym4', ...
                        'DenoisingMethod','Bayes', ...
                        'ThresholdRule','Median');
            end

            if strcmpi(axis_kind,'mel')
                S = melSpectrogram(x, Fs, ...
                        'Window', window, ...
                        'OverlapLength', overlap, ...
                        'FFTLength', fftlen, ...
                        'NumBands', 128);
            else
                % output arguments requested, so nothing is plotted
                S = abs(spectrogram(x, window, overlap, fftlen, Fs));
            end

            S = log10(S + 1e-6);
            S = mat2gray(S);
            S = imresize(S, [224 224]);

            imwrite(S, fullfile(out_folder, strrep(files(i).name,'.wav','.png')));
        end

        fprintf('  %-6s %-10s %5d images\n', sets{s}, classes{c}, numel(files));
    end
end
end
