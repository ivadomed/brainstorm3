function tutorial_resting(tutorial_dir)
% TUTORIAL_RESTING: Script that reproduces the results of the online tutorial "Resting state MEG".
%
% DESCRIPTION:
%     This tutorial is based on two blocks of 10 minutes of resting state eyes open recorded 
%     at the Montreal Neurological Institute in 2011 with a CTF MEG 275 system. 
%
% INPUTS: 
%     tutorial_dir: Directory where the sample_resting.zip file has been unzipped

% @=============================================================================
% This function is part of the Brainstorm software:
% https://neuroimage.usc.edu/brainstorm
% 
% Copyright (c)2000-2019 University of Southern California & McGill University
% This software is distributed under the terms of the GNU General Public License
% as published by the Free Software Foundation. Further details on the GPLv3
% license can be found at http://www.gnu.org/copyleft/gpl.html.
% 
% FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE
% UNIVERSITY OF SOUTHERN CALIFORNIA AND ITS COLLABORATORS DO NOT MAKE ANY
% WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY
% LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
%
% For more information type "brainstorm license" at command prompt.
% =============================================================================@
%
% Author: Francois Tadel, 2013-2016


% ===== FILES TO IMPORT =====
% You have to specify the folder in which the tutorial dataset is unzipped
if (nargin == 0) || isempty(tutorial_dir) || ~file_exist(tutorial_dir)
    error('The first argument must be the full path to the tutorial dataset folder.');
end
% Build the files name of the files to import
AnatDir    = fullfile(tutorial_dir, 'sample_resting', 'Anatomy');
Run1File   = fullfile(tutorial_dir, 'sample_resting', 'Data', 'subj002_spontaneous_20111102_01_AUX.ds');
Run2File   = fullfile(tutorial_dir, 'sample_resting', 'Data', 'subj002_spontaneous_20111102_02_AUX.ds');
NoiseFile  = fullfile(tutorial_dir, 'sample_resting', 'Data', 'subj002_noise_20111104_02.ds');
Event1File = fullfile(Run1File, 'events_BAD.mat');
Event2File = fullfile(Run2File, 'events_BAD.mat');
% Check if the folder contains the required files
if ~file_exist(Run1File)
    error(['The folder ' tutorial_dir ' does not contain the folder from the file sample_resting.zip.']);
end

% ===== CREATE PROTOCOL =====
% The protocol name has to be a valid folder name (no spaces, no weird characters...)
ProtocolName = 'TutorialResting';
% Start brainstorm without the GUI
if ~brainstorm('status')
    brainstorm nogui
end
% Delete existing protocol
gui_brainstorm('DeleteProtocol', ProtocolName);
% Create new protocol
gui_brainstorm('CreateProtocol', ProtocolName, 0, 0);
% Start a new report
bst_report('Start');


% ===== ANATOMY =====
% Subject name
SubjectName = 'Subject02';
% Process: Import FreeSurfer folder
bst_process('CallProcess', 'process_import_anatomy', [], [], ...
    'subjectname', SubjectName, ...
    'mrifile',     {AnatDir, 'FreeSurfer'}, ...
    'nvertices', 15000, ...
    'nas', [128, 225, 135], ...
    'lpa', [ 54, 115, 107], ...
    'rpa', [204, 115,  99]);


% ===== LINK CONTINUOUS FILE =====
% Process: Create link to raw file (Run1)
sFileRun1 = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
    'subjectname',  SubjectName, ...
    'datafile',     {Run1File, 'CTF'}, ...
    'channelalign', 1);

% Process: Events: Import from file (Run1)
bst_process('CallProcess', 'process_evt_import', sFileRun1, [], ...
    'evtfile', {Event1File, 'BST'});

% Link noise recordings (Noise)
sFileNoise = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
    'subjectname',  SubjectName, ...
    'datafile',     {NoiseFile, 'CTF'}, ...
    'channelalign', 0);

% Process: Snapshot: Sensors/MRI registration (Run1)
bst_process('CallProcess', 'process_snapshot', sFileRun1, [], ...
    'target',   1, ...  % Sensors/MRI registration
    'modality', 1, ...  % MEG (All)
    'orient',   1, ...  % left
    'Comment',  'MEG/MRI Registration');

% Process: Create link to raw file
% sFilesRun2 = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
%     'subjectname',  SubjectName, ...
%     'datafile',     {Run2File, 'CTF'}, ...
%     'channelalign', 1);
% Process: Events: Import from file
% bst_process('CallProcess', 'process_evt_import', sFilesRun2, [], ...
%     'evtfile', {Event2File, 'BST'});


% ===== CORRECT BLINKS AND HEARTBEATS =====
% Process: Detect heartbeats (Run1)
sFileRun1 = bst_process('CallProcess', 'process_evt_detect_ecg', sFileRun1, [], ...
    'channelname', 'EEG057', ...
    'timewindow',  [], ...
    'eventname',   'cardiac');

% Process: Detect eye blinks (Run1)
sFileRun1 = bst_process('CallProcess', 'process_evt_detect_eog', sFileRun1, [], ...
    'channelname', 'EEG058', ...
    'timewindow',  [], ...
    'eventname',   'blink');

% Process: Remove simultaneous (Run1)
sFileRun1 = bst_process('CallProcess', 'process_evt_remove_simult', sFileRun1, [], ...
    'remove', 'cardiac', ...
    'target', 'blink', ...
    'dt',     0.25, ...
    'rename', 0);

% Process: SSP ECG: cardiac (Run1)
sFileRun1 = bst_process('CallProcess', 'process_ssp_ecg', sFileRun1, [], ...
    'eventname',   'cardiac', ...
    'sensortypes', 'MEG', ...
    'usessp',      1, ...
    'select',      1);

% Process: SSP EOG: blink (Run1)
sFileRun1 = bst_process('CallProcess', 'process_ssp_eog', sFileRun1, [], ...
    'eventname',   'blink', ...
    'sensortypes', 'MEG', ...
    'usessp',      1, ...
    'select',      1);

% Process: Snapshot: SSP projectors (Run1)
bst_process('CallProcess', 'process_snapshot', sFileRun1, [], ...
    'target',  2, ...  % SSP projectors
    'Comment', 'SSP projectors');


% ===== SOURCE MODELING =====
% Process: Compute head model (Run1)
bst_process('CallProcess', 'process_headmodel', sFileRun1, [], ...
    'comment', '', ...
    'sourcespace', 1, ...
    'meg',         3);  % Overlapping spheres

% Process: Compute noise covariance (Noise)
bst_process('CallProcess', 'process_noisecov', sFileNoise, [], ...
    'baseline',     [], ...
    'sensortypes',  'MEG', ...  % Noise covariance     (covariance over baseline time window)
    'dcoffset',     1, ...
    'identity',     0, ...
    'copycond',     1, ...
    'copysubj',     0, ...
    'replacefile',  1);  % Replace

% Process: Snapshot: Noise covariance
bst_process('CallProcess', 'process_snapshot', sFileNoise, [], ...
    'target',  3, ...  % Noise covariance
    'Comment', 'Noise covariance');

% Process: Compute sources (Run1)
sFileRun1Source = bst_process('CallProcess', 'process_inverse', sFileRun1, [], ...
    'comment', '', ...
    'method',  1, ...  % Minimum norm estimates (wMNE)
    'wmne',    struct(...
         'NoiseCov',      [], ...
         'InverseMethod', 'wmne', ...
         'ChannelTypes',  {{}}, ...
         'SNR',           3, ...
         'diagnoise',     0, ...
         'SourceOrient',  {{'fixed'}}, ...
         'loose',         0.2, ...
         'depth',         1, ...
         'weightexp',     0.5, ...
         'weightlimit',   10, ...
         'regnoise',      1, ...
         'magreg',        0.1, ...
         'gradreg',       0.1, ...
         'eegreg',        0.1, ...
         'ecogreg',       0.1, ...
         'seegreg',       0.1, ...
         'fMRI',          [], ...
         'fMRIthresh',    [], ...
         'fMRIoff',       0.1, ...
         'pca',           1), ...
    'sensortypes', 'MEG', ...
    'output',      2);  % Kernel only: one per file

% ===== RESTING STATE PIPELINE =====
% Process: Phase-amplitude coupling
sFilePac = bst_process('CallProcess', 'process_pac', sFileRun1Source, [], ...
    'timewindow',     [400, 600], ...
    'scouts',         {'Brodmann', {'V1 R'}}, ...
    'scoutfunc',      1, ...  % Mean
    'scouttime',      1, ...  % Before
    'nesting',        [2, 14], ...
    'nested',         [40, 150], ...
    'numfreqs',       0, ...
    'parallel',       0, ...
    'ismex',          1, ...
    'max_block_size', 1, ...
    'avgoutput',      0, ...
    'savemax',        0);

% Process: Canolty maps
sFileCanolty = bst_process('CallProcess', 'process_canoltymap', sFileRun1Source, [], ...
    'timewindow',     [400, 600], ...
    'scouts',         {'Brodmann', {'V1 R'}}, ...
    'scoutfunc',      5, ...  % All
    'scouttime',      2, ...
    'epochtime',      [-0.5, 0.5], ...
    'lowfreq',        11, ...
    'max_block_size', 100, ...
    'save_erp',       1);

% Process: Canolty maps (FileB=MaxPAC)
sFileCanolty2 = bst_process('CallProcess', 'process_canoltymap2', sFileRun1Source, sFilePac, ...
    'timewindow',     [400, 600], ...
    'scouts',         {'Brodmann', {'V1 R'}}, ...
    'scoutfunc',      5, ...  % All
    'scouttime',      1, ...
    'epochtime',      [-0.5, 0.5], ...
    'max_block_size', 100, ...
    'save_erp',       1);

% Display results
hFig1 = view_pac(sFilePac.FileName);
hFig2 = view_timefreq(sFileCanolty.FileName);
hFig3 = view_timefreq(sFileCanolty2.FileName);
pause(1);
close([hFig1 hFig2 hFig3]);

% Save and display report
ReportFile = bst_report('Save', sFileCanolty2);
bst_report('Open', ReportFile);





