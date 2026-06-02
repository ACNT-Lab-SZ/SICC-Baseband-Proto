function startup_ntn_otfs_project()
%STARTUP_NTN_OTFS_PROJECT Configure MATLAB paths for the NTN OTFS project.

projectRoot = fileparts(fileparts(mfilename('fullpath')));

addpath(projectRoot);
addpath(fullfile(projectRoot, 'config'));
addpath(fullfile(projectRoot, 'examples'));
addpath(fullfile(projectRoot, 'tests'));
addpath(genpath(fullfile(projectRoot, 'src')));

syncRoot = fullfile(projectRoot, 'Synchronization_MATLAB');
if isfolder(syncRoot)
    addpath(genpath(syncRoot));
end

codeRoot = fullfile(projectRoot, 'Code_Matrices_Lib');
if isfolder(codeRoot)
    addpath(genpath(codeRoot));
end

fprintf('NTN OTFS project paths configured from %s\n', projectRoot);
end
