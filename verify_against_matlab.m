function verify_against_matlab()
% VERIFY_AGAINST_MATLAB  Check the website's data against the real pipeline.
%
% The site ships a feature matrix (site/data/features_45.bin) built by a Python
% port of feature_label_moving_bin.m, and trains it with a JavaScript port of
% run_lda_auc.m. Ports drift. This script is how you find out.
%
% It feeds the website's own feature matrix to the *real* run_lda_auc.m and
% compares the result with the AUCs committed in outputs/LDA/LDA_results_raw.mat.
% If the Python feature extraction is faithful, these agree to ~3 decimals,
% because run_lda_auc seeds its RNG (rng(0,'twister')) and so is deterministic.
%
% Run it from anywhere:
%
%     >> setup_paths
%     >> verify_against_matlab
%
% Last run (2026-07-15, R2025a): all three tasks matched exactly.
%
% If this FAILS, the likely cause is that feature_label_moving_bin.m changed and
% site/build_data.py was not updated to match. Re-run:
%
%     python3 site/build_data.py
%
% See also: site/_selftest.html, which does the equivalent check for the
% JavaScript classifier from inside a browser.

    here = fileparts(mfilename('fullpath'));
    repo = fileparts(here);
    DEMO = 45;

    binFile  = fullfile(here, 'data', 'features_45.bin');
    metaFile = fullfile(here, 'data', 'features_45.json');
    refFile  = fullfile(repo, 'outputs', 'LDA', 'LDA_results_raw.mat');

    for f = {binFile, metaFile, refFile}
        assert(isfile(f{1}), 'Missing %s', f{1});
    end
    assert(exist('run_lda_auc', 'file') == 2, ...
        'run_lda_auc not found -- run setup_paths first.');

    % --- read the website's feature matrix -------------------------------
    meta = jsondecode(fileread(metaFile));
    fid = fopen(binFile, 'r');
    raw = fread(fid, inf, 'int16=>double');
    fclose(fid);

    nTrials = meta.shape(1);
    nFeat   = meta.shape(2);
    assert(numel(raw) == nTrials * nFeat, 'features_45.bin is the wrong size');

    % the file is row-major (C order); MATLAB reshapes column-major, so transpose
    X = reshape(raw, nFeat, nTrials).' * meta.scale;
    y = double(meta.labels(:));

    fprintf('\nWebsite feature matrix: %d trials x %d features (participant %d)\n', ...
        nTrials, nFeat, meta.participant);
    fprintf('Trial codes: %d hits, %d misses, %d CR, %d FA\n\n', ...
        sum(y == 1), sum(y == 2), sum(y == 3), sum(y == 4));

    % --- the committed reference -----------------------------------------
    R = load(refFile);
    ref = R.AUC_all.(sprintf('P%02d', DEMO));

    tasks = { ...
        'OLDvsNew', 'OldNew',  @(X, y) deal(X,               ismember(y, [1 2])); ...
        'HitMiss',  'HitMiss', @(X, y) deal(X(ismember(y, [1 2]), :), y(ismember(y, [1 2])) == 1); ...
        'FAvsCR',   'FAvsCR',  @(X, y) deal(X(ismember(y, [3 4]), :), y(ismember(y, [3 4])) == 4)};

    fprintf('%-10s %10s %10s %8s\n', 'task', 'MATLAB', 'committed', 'diff');
    fprintf('%s\n', repmat('-', 1, 42));

    worst = 0;
    for k = 1:size(tasks, 1)
        [Xt, yb] = tasks{k, 3}(X, y);
        auc = run_lda_auc(Xt, double(yb), tasks{k, 1});
        expected = ref.(tasks{k, 2});
        d = abs(auc - expected);
        worst = max(worst, d);
        fprintf('%-10s %10.4f %10.4f %8.4f\n', tasks{k, 1}, auc, expected, d);
    end

    fprintf('%s\n', repmat('-', 1, 42));
    if worst < 0.005
        fprintf('PASS  worst difference %.4f -- the Python feature port is faithful.\n\n', worst);
    else
        fprintf(2, ['FAIL  worst difference %.4f.\n' ...
            'The website''s features no longer match this pipeline.\n' ...
            'Re-run:  python3 site/build_data.py\n\n'], worst);
    end
end
