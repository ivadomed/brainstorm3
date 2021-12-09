function varargout = process_ivadomed_create_dataset( varargin )
% PROCESS_IVADOMED_CREATE_DATASET: this function converts trials to NIFTI files in BIDS
% format to be used in the IVADOMED fraework for training deep learning models
% https://ivadomed.org/en/latest/index.html

% USAGE:    sProcess = process_ivadomed_create_dataset('GetDescription')
%        OutputFiles = process_ivadomed_create_dataset('Run', sProcess, sInput)
%        OutputFiles = process_ivadomed_create_dataset('Run', sProcess, sInput, return_filenames)
%                     return_filenames: Flag to return the filenames of the
%                     created dataset
% @=============================================================================
% This function is part of the Brainstorm software:
% https://neuroimage.usc.edu/brainstorm
% 
% Copyright (c)2000-2020 University of Southern California & McGill University
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
% Author: Konstantinos Nasiotis, 2021

eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<*DEFNU>
    % Description the process
    sProcess.Comment     = 'Ivadomed Create BIDS Dataset';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3112;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data'};
    sProcess.OutputTypes = {'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    
    % Modality Selection
    sProcess.options.label11.Comment = '<BR><B>Modality selection:</B>';
    sProcess.options.label11.Type    = 'label';
    sProcess.options.modality.Comment = {'MEG', 'EEG', 'MEG+EEG', 'fNIRS'};
    sProcess.options.modality.Type    = 'radio';
    sProcess.options.modality.Value   = 1;
    
    % BIDS
    sProcess.options.label1.Comment = '<B>BIDS conversion parameters:</B>';
    sProcess.options.label1.Type    = 'label';
    % Event name
    sProcess.options.eventname.Comment = 'Event for ground truth';
    sProcess.options.eventname.Type    = 'text';
    sProcess.options.eventname.Value   = 'event1';
    % Event help comment
    sProcess.options.eventname_help.Comment = '<I><FONT color="#777777">If the eventname is left empty, the annotations are based on the following time-window within each trial</FONT></I>';
    sProcess.options.eventname_help.Type    = 'label';
    % Options: Segment around spike
    sProcess.options.timewindow.Comment  = 'Annotations Time window: ';
    sProcess.options.timewindow.Type     = 'range';
    sProcess.options.timewindow.Value    = {[-0.150, 0.150],'ms',[]};
    % Event help comment
    sProcess.options.timewindow_help.Comment = '<I><FONT color="#777777">This time window is only used for annotating around single events or if event name is empty</FONT></I>';
    sProcess.options.timewindow_help.Type    = 'label';
    
    % Needed Fs
    sProcess.options.fs.Comment = 'Resampling rate <I><FONT color="#777777">(empty for no resampling)</FONT></I>';
    sProcess.options.fs.Type    = 'value';
    sProcess.options.fs.Value   = {100, 'Hz', 0};
    
    % Needed Jitter for cropping
    sProcess.options.jitter.Comment = 'Jitter value';
    sProcess.options.jitter.Type    = 'value';
    sProcess.options.jitter.Value   = {200, 'ms', 0};
    % Jitter comment
    sProcess.options.jitter_help.Comment = '<I><FONT color="#777777">This is used to crop the edges of each trial so the trained model doesn"t learn the position of the event</FONT></I>';
    sProcess.options.jitter_help.Type    = 'label';
    
    % Data augmentation parameters
    sProcess.options.label15.Comment = '<BR><B>Data augmentation parameters (signals level):</B>';
    sProcess.options.label15.Type    = 'label';
    % Use gaussian soft annotation
    sProcess.options.channelDropOut.Comment = 'Channels drop-out';
    sProcess.options.channelDropOut.Type    = 'value';
    sProcess.options.channelDropOut.Value   = {[], 'channels', 0};
    sProcess.options.channelDropOut_help.Comment = '<I><FONT color="#777777">Remove n channels randomly up to the selected value</FONT></I>';
    sProcess.options.channelDropOut_help.Type = 'label';
    
    % Annotation Options
    sProcess.options.label14.Comment = '<BR><B>Annotation Parameters:</B>';
    sProcess.options.label14.Type    = 'label';
    % Use gaussian soft annotation
    sProcess.options.gaussian_annot.Comment = 'Gaussian annotation';
    sProcess.options.gaussian_annot.Type    = 'checkbox';
    sProcess.options.gaussian_annot.Value   = 0;
    sProcess.options.gaussian_annot_help.Comment = '<I><FONT color="#777777">The annotation within the annotation window will be a gaussian function</FONT></I>';
    sProcess.options.gaussian_annot_help.Type = 'label';
    
    % Needed threshold for soft annotation
    sProcess.options.annotthresh.Comment = 'Soft annotation threshold (0,1)';
    sProcess.options.annotthresh.Type    = 'value';
    sProcess.options.annotthresh.Value   = {[], [], 1};
    % Annotation threshold comment
    sProcess.options.annotthresh_help.Comment = '<I><FONT color="#777777">If selected, the annotation will have a soft threshold. Leave empty for hard annotation at 0.5 during inference</FONT></I>';
    sProcess.options.annotthresh_help.Type    = 'label';
    % Parallel processing
    sProcess.options.paral.Comment = 'Parallel processing';
    sProcess.options.paral.Type    = 'checkbox';
    sProcess.options.paral.Value   = 0;
    % Method: BIDS subject selection
    sProcess.options.label2.Comment = '<B>BIDS folders creation </B>';
    sProcess.options.label2.Type    = 'label';
    sProcess.options.bidsFolders.Comment = {'Normal', 'Separate runs/sessions as different subjects', 'Separate each trial as different subjects'};
    sProcess.options.bidsFolders.Type    = 'radio';
    sProcess.options.bidsFolders.Value   = 1;
    
    % Conversion of both trials and their derivatives
    sProcess.options.convert.Type   = 'text';
    sProcess.options.convert.Value  = 'conversion';  % Other option: 'segmentation'
    sProcess.options.convert.Hidden = 1;
    
    % Display example image in FSLeyes
    sProcess.options.label3.Comment = '<B>FSLeyes </B>';
    sProcess.options.label3.Type    = 'label';
    sProcess.options.dispExample.Comment = 'Open an example image/derivative on FSLeyes';
    sProcess.options.dispExample.Type    = 'checkbox';
    sProcess.options.dispExample.Value   = 0;
    
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess)
    Comment = 'Ivadomed - Create BIDS dataset';
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs, return_filenames)

    % CHECK IF IVADOMED EXISTS
    fname = bst_fullfile(bst_get('UserPluginsDir'), 'ivadomed', 'ivadomed-master', 'ivadomed', 'main.py');
    if ~(exist(fname, 'file') == 2)
        
        % Check if ivadomed can be accessed from a system call
        % in case the user installed it outside of Brainstorm
        output = system('ivadomed -h');
        if output~=0
            OutputFiles = {};
            bst_report('Error', sProcess, sInputs, 'Ivadomed package is not accessible. Are you running Matlab through an anaconda environment that has Ivadomed installed?');
            return
        end
    end
    
    
    %% Close all existing figures
    bst_memory('UnloadAll', 'Forced');

    
    %% Sampling rate to be used when creating the NIFTI files
    wanted_Fs = sProcess.options.fs.Value{1};
    
    %% Modality selected
    
    modality = sProcess.options.modality.Comment{sProcess.options.modality.Value};
    
    %% Do some checks on the parameters
    
    if strcmp(sProcess.options.convert.Value, 'conversion')
    
        % In case the selected event is single event, or the eventname isempty,
        % make sure the time-window has values
        inputs_to_remove = false(length(sInputs),1);

        runs = cell(length(sInputs),1);

        for iInput = 1:length(sInputs)
            dataMat = in_bst(sInputs(iInput).FileName, 'Events');
            events = dataMat.Events;

            Time = in_bst(sInputs(iInput).FileName, 'Time');
            
            % Gather all runs that are used -  THIS IS FOR BIDS DATASETS
            % ONLY - TODO
            splits = split(sInputs(iInput).Condition, '_');
            run = splits(contains(splits, 'run')); splitsRun = split(run,'-');
            iRun = str2double(splitsRun{2});

            runs{iInput} = num2str(iRun);

            if isempty(sProcess.options.eventname.Value)
                if length(sProcess.options.timewindow.Value{1})<2 || isempty(sProcess.options.timewindow.Value{1}) || ...
                        sProcess.options.timewindow.Value{1}(1)<Time(1) || sProcess.options.timewindow.Value{1}(2)>Time(end)
                    inputs_to_remove(iInput) = true;
                    bst_report('Warning', sProcess, sInputs, ['The time window selected for annotation is not within the Time segment of trial: ' dataMat.Comment  ' . Ignoring this trial']);
                end
            else
                if isfield(events, 'label')
                    [isSelectedEventPresent, index] = ismember(sProcess.options.eventname.Value, {events.label});
                else
                    isSelectedEventPresent = false;
                end

                if ~isSelectedEventPresent
                    inputs_to_remove(iInput) = true;
                    bst_report('Warning', sProcess, sInputs, ['The selected event does not exist within trial: ' dataMat.Comment ' . Ignoring this trial']);
                else
%                     % TODO - THIS IS WRONG - IT SHOULD BE REJECTED IF ALL ANNOTATIONS ARE OUT OF BOUNDS, NOT JUST THE FIRST AND LAST
                    if Time(1) > sProcess.options.timewindow.Value{1}(1) + events(index).times(1) || Time(end) < events(index).times(end) + sProcess.options.timewindow.Value{1}(2)
                        inputs_to_remove(iInput) = true;
                        bst_report('Warning', sProcess, sInputs, ['The time window selected for annotation is not within the Time segment of trial: ' dataMat.Comment ' . Ignoring this trial']);                    
                    end
%                     continue

                end
            end
        end

        % The following works on Matlab R2021a - I think older versions
        % need another command to squeeze the empty structure entries - TODO
        sInputs(inputs_to_remove) = [];
        runs = unique(runs);
    end
    
    
    if isempty(sInputs)
        bst_report('Error', sProcess, sInputs, 'No inputs selected');
        OutputFiles = {};
        return
    end
    
    %% Gather F and filenames for all files here - This shouldn't create a memory issue
    % Reason why all of them are imported here is that in_bst doesn't like
    % to be parallelized (as far as I checked), so can't call it within 
    % convertTopography2matrix
    
    
    
    protocol = bst_get('ProtocolInfo');
    if strcmp(sProcess.options.convert.Value, 'conversion')
        % TODO - THIS IS HARDCODED FOR BIDS DATASETS TO DISPLAY THE RUNS THAT
        % WERE USED - REMOVE FOR FINAL PRODUCTION
        runs_string = join(runs,'-');
        parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '_' modality '_runs_' runs_string{1}]);
    else
        parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       protocol.Comment);
    end
    
    
    empty_IVADOMED_folder = true;
    % Remove previous entries
    if empty_IVADOMED_folder
        if exist(parentPath, 'dir') 
            rmdir(parentPath, 's')
        end
        if exist([parentPath '-meta'], 'dir') 
            rmdir([parentPath '-meta'], 's')
        end
    end
                   
    channels_times_path = [parentPath '-meta'];
                   
    % Differentiate simple conversion and segementation paths
    if strcmp(sProcess.options.convert.Value, 'segmentation')
        channels_times_path = [channels_times_path '-segmentation'];
        parentPath = [parentPath '-segmentation'];
    end
    
        
    if isempty(sProcess.options.eventname.Value)
        annotation = 'centered';
    else
        annotation = sProcess.options.eventname.Value;
    end
                   
    % Hack to accommodate ivadomed derivative selection:
    % https://github.com/ivadomed/ivadomed/blob/master/ivadomed/loader/utils.py # L812
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'; % This hack accommodates up to 240 trials within a run - for more find another solution 
                                            % - like double letters (not the same though or the same IVADOMED loader problem would occur)
                 
    ii = 1;
                                            
    info_trials = struct;
    
    txt  = [];
    
    for iInput = 1:length(sInputs)
        info_trials(iInput).FileName = sInputs(iInput).FileName;
        info_trials(iInput).subject = lower(str_remove_spec_chars(sInputs(iInput).SubjectName));
        info_trials(iInput).session = lower(str_remove_spec_chars(sInputs(iInput).Condition));
        
        % The trial # needs special attention
        splitComment = split(sInputs(iInput).Comment,{'(#',')'});
        comment = lower(str_remove_spec_chars(splitComment{1}));
        iEpoch = str2double(splitComment{2});
        
        if ~strcmp(modality, 'MEG+EEG')
            if ~sProcess.options.bidsFolders.Value==3
            
                % This is a hack until the ivadomed code is changed - TODO
                iLetter = floor(iEpoch/10);
                if iLetter == 0
                    iEpoch = num2str(iEpoch);
                elseif iLetter <= 24
                    iEpoch = [letters(iLetter) num2str(iEpoch)];
                elseif iLetter > 24 && iLetter <= 48
                    iEpoch = [letters(iLetter-24) 'A' num2str(iEpoch)];
                elseif iLetter > 48 && iLetter < 72
                    iEpoch = [letters(iLetter-48) 'B' num2str(iEpoch)];
                end
                info_trials(iInput).trial = {[modality comment iEpoch]};
            else
                info_trials(iInput).trial = {[modality comment]};
            end
        else
            info_trials(iInput).trial = {'MEG', 'EEG'};
        end
            
        % Load data structure
        info_trials(iInput).dataMat = in_bst(sInputs(iInput).FileName);
        
        
        % And resample if needed
        current_Fs = round(1/diff(info_trials(iInput).dataMat.Time(1:2)));
        if ~isnan(wanted_Fs) && current_Fs~=wanted_Fs
            %[x, time_out] = process_resample('Compute', x, time_in, NewRate)
            [info_trials(iInput).dataMat.F, info_trials(iInput).dataMat.Time] = process_resample('Compute', info_trials(iInput).dataMat.F, info_trials(iInput).dataMat.Time, wanted_Fs);
        else
            wanted_Fs = current_Fs;  % This is done here so it can be saved on the saved config file, and ultimately this sampling rate be used on the segementation function
        end

          % Get output study
        [tmp, iStudy] = bst_process('GetOutputStudy', sProcess, sInputs(iInput));
        % Get channel file
        sChannel = bst_get('ChannelForStudy', iStudy);
        % Load channel file
        info_trials(iInput).ChannelMat = in_bst_channel(sChannel.FileName);
        
        % Use default anatomy (IS THIS SOMETHING IMPORTANT TO CONSIDER CHANGING - MAYBE FOR SOURCE LOCALIZATION ON MEG STUDIES???)
        % TODO - CONSIDER ADDING THE INDIVIDUAL ANATOMY HERE
        info_trials(iInput).sMri = load(bst_fullfile(bst_get('BrainstormHomeDir'), 'defaults', 'anatomy', 'ICBM152', 'subjectimage_T1.mat'));
        info_trials(iInput).parentPath = parentPath;
        info_trials(iInput).channels_times_path = channels_times_path;
        
        
        % Get output filename
        subject = info_trials(iInput).subject;
        session = info_trials(iInput).session;
        trial   = info_trials(iInput).trial;
        
        for iTrial = 1:length(trial)
        
            if sProcess.options.bidsFolders.Value==1
                % Images
                info_trials(iInput).OutputMriFile      = bst_fullfile(['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial{iTrial} '.nii']);
                info_trials(iInput).OutputChannelsFile = bst_fullfile(['sub-' subject], ['ses-' session], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFile    = bst_fullfile(['sub-' subject], ['ses-' session], 'anat', ['times_' trial{iTrial} '.csv']);

                % Derivatives
                info_trials(iInput).OutputMriFileDerivative      = bst_fullfile('derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial{iTrial} '_' annotation '.nii']);
                info_trials(iInput).OutputChannelsFileDerivative = bst_fullfile('derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFileDerivative    = bst_fullfile('derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['times_' trial{1} '.csv']);

            elseif sProcess.options.bidsFolders.Value==2
                % Images
                subject = [subject session];
                info_trials(iInput).OutputMriFile      = bst_fullfile(['sub-' subject], 'anat', ['sub-' subject '_' trial{iTrial} '.nii']);
                info_trials(iInput).OutputChannelsFile = bst_fullfile(['sub-' subject], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFile    = bst_fullfile(['sub-' subject], 'anat', ['times_' trial{iTrial} '.csv']);

                % Derivatives
                info_trials(iInput).OutputMriFileDerivative{iTrial} = bst_fullfile('derivatives', 'labels', ['sub-' subject], 'anat', ['sub-' subject '_' trial{iTrial} '_' annotation '.nii']);
                info_trials(iInput).OutputChannelsFileDerivative    = bst_fullfile('derivatives', 'labels', ['sub-' subject], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFileDerivative       = bst_fullfile('derivatives', 'labels', ['sub-' subject], 'anat', ['times_' trial{1} '.csv']);
                
            elseif sProcess.options.bidsFolders.Value==3
                % Images
                if ismember(info_trials(iInput).trial, {'MEG', 'EEG'})
                    subject_new = [subject session comment num2str(iInput)];
                else
                    subject_new = [subject session num2str(iInput)];
                end
                info_trials(iInput).subject            = subject_new;
                info_trials(iInput).OutputMriFile{iTrial}      = bst_fullfile(['sub-' subject_new], 'anat', ['sub-' subject_new '_' trial{iTrial} '.nii']);
                info_trials(iInput).OutputChannelsFile = bst_fullfile(['sub-' subject_new], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFile    = bst_fullfile(['sub-' subject_new], 'anat', ['times_' trial{1} '.csv']);

                % Derivatives
                info_trials(iInput).OutputMriFileDerivative{iTrial}   = bst_fullfile('derivatives', 'labels', ['sub-' subject_new], 'anat', ['sub-' subject_new '_' trial{iTrial} '_' annotation '.nii']);
                info_trials(iInput).OutputChannelsFileDerivative      = bst_fullfile('derivatives', 'labels', ['sub-' subject_new], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFileDerivative{iTrial} = bst_fullfile('derivatives', 'labels', ['sub-' subject_new], 'anat', ['times_' trial{1} '.csv']);

                if iInput==1
                    txt = ['"' trial{iTrial} '"'];
                else
                    txt = [txt ', "' trial{iTrial} '"'];
                end
            end
        end
    end
    
    % Keep only unique entries to be used in the Ivadomed config.json file
    % This is for the entries on "contrast_params": {"training_validation"}
    % and "contrast_params": {"testing"}
    txt = split(txt, ', '); 
    txt = unique(txt); 
    contrast_params_txt = txt;
    txt = strcat(txt);
    txt = join(txt, ', '); 
    txt = txt{1};
    
    
    disp(['Entries for "contrast_params": {training_validation} {testing}: ' txt])
    
    %% Open a figure window to inherit properties
    
%     [hFig, iDS, iFig] = view_topography(sInput.FileName, 'MEG', '2DSensorCap');        
%     [hFig, iFig, iDS] = bst_figures('GetFigure', GlobalData.DataSet.Figure.hFigure);
%     set(hFig, 'Visible', 'off');
%     
%     
%     bst_get('Layout', 'WindowManager')
%     
    figures_struct = struct('FigureObject',[], 'Status', [], 'Trialfilenames', {sInputs.FileName});
    
    
    %% Convert the input trials to NIFTI files
    start_time = tic;
    
    % If MEG and EEG is requested simultaneously, create a NIFTI file for each
    if strcmp(modality, 'MEG+EEG')
        filenames = cell(length(info_trials)*2,1);
        subjects = cell(length(info_trials)*2,1);
        
        bst_progress('start', 'Ivadomed', 'Converting trials to NIFTI files...', 0, length(sInputs));
        for iFile = 1:length(info_trials)
            disp(['Trial: ' num2str(iFile) '/' num2str(length(info_trials))])
            [filenames(2*(iFile-1)+1:2*iFile), subjects(2*(iFile-1)+1:2*iFile)] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
            bst_progress('inc', 1);
        end
    else
        filenames = cell(length(info_trials),1);
        subjects = cell(length(info_trials),1);
        
        bst_progress('start', 'Ivadomed', 'Converting trials to NIFTI files...', 0, length(sInputs));
        for iFile = 1:length(info_trials)
            disp(['Trial: ' num2str(iFile) '/' num2str(length(info_trials))])
            [filenames(iFile), subjects(iFile)] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
            bst_progress('inc', 1);
        end
    end
    
    disp(['Total time for converting ' num2str(length(info_trials)) ' trials: ' num2str(toc(start_time)) ' seconds'])
    
    
    % Return the filenames only when process_ivadomed_segmentation calls
    % them - Don't get them by default
    if ~exist('return_filenames','var')
        OutputFiles = {};
    else
        if return_filenames==1
            OutputFiles = filenames;
        else
            OutputFiles = {};
        end
    end
        
    % === EXPORT BIDS FILES ===
    export_participants_tsv(parentPath, unique(subjects))
    export_participants_json(parentPath)
    export_dataset_description(parentPath)
    export_readme(parentPath)
    
    % === CREATE A TEMPLATE CONFIG.JSON WITH THE CREATED CONTRAST
    % PARAMETERS - REEVALUATE IF THIS IS NEEDED FOR FINAL RELEASE
    if strcmp(sProcess.options.convert.Value, 'conversion')
        modify_config_json(parentPath, modality, annotation, contrast_params_txt, sProcess, {sInputs.FileName})
    end
    
    
    % === OPEN EXAMPLE IMAGE/DERIVATIVE IN FSLEYES ===
    if sProcess.options.dispExample.Value
        % Check if FSLeyes is installed in the Conda environment
        if ismac
            output=1; % NOT TESTED ON MAC YET -TODO
        elseif isunix
            output = system('/home/nas/anaconda3/envs/ivadomed/bin/fsleyes -h');
            OutputMriFile = bst_fullfile(info_trials(1).parentPath, info_trials(1).OutputMriFile{1});
            OutputMriFileDerivative = bst_fullfile(info_trials(1).parentPath, info_trials(1).OutputMriFileDerivative{1});
            command_to_run = ['/home/nas/anaconda3/envs/ivadomed/bin/fsleyes ' OutputMriFile '.gz -cm render3 ' OutputMriFileDerivative '.gz -cm green --alpha 60 &' ];
        elseif ispc
            output=1; % NOT TESTED ON WINDOWS YET -TODO
        else
            disp('Platform not supported')
        end
        if output~=0
            bst_report('Warning', sProcess, sInputs(1), 'Fsleyes package is not accessible. Are you running Matlab through an anaconda environment that has Fsleyes installed?');
            disp(command_to_run)
            return
        else
            system(command_to_run);
        end
    end
    
    [temp1,folder_output,temp2] = bst_fileparts(parentPath);

    % Command to run on the terminal for copying files
    disp(['scp -rp ' parentPath ' u111358@rosenberg.neuro.polymtl.ca:/home/GRAMES.POLYMTL.CA/u111358/data_nvme_u111358/EEG-ivado/epilepsy/distant_spikes_2_seconds_gaussian_annotation/data_' folder_output '_gaussian_annotation']);
    
end


function [OutputMriFile, subject] = convertTopography2matrix(single_info_trial, sProcess, iFile, figures_struct)
%   % Ignoring the bad sensors in the interpolation, so some values will be interpolated from the good sensors
%   WExtrap = GetInterpolation(iDS, iFig, TopoInfo, Vertices, Faces, bfs_center, bfs_radius, chan_loc(selChan,:));
% 
%         
%   [DataToPlot, Time, selChan, overlayLabels, dispNames, StatThreshUnder, StatThreshOver] = GetFigureData(iDS, iFig, 0);

    % TODO
    disp('DOES The colormap need to be GRAY???') % ANSWER: YES
        
   
        %% ADDING TEST FOR PARTIAL ANNOTATION
        
        
        
        
        
        
        fake_partial_annotation = 0;
        
        
        randomize_annotation_position_for_some_channels = 0;
%         
%         
%         if strcmp(sProcess.options.convert.Value, 'conversion')
%         
%         
%         
%             eyes_channel_Names = {'MLT21', 'MLT31','MLT32', 'MLT41', 'MLT51', 'MLT42', 'MLT14', 'MLT25', 'MRT21', 'MRT31', 'MRT32', 'MRT41', 'MRF14', 'MRF25', 'MRT51', 'MRT42'};
% 
%             if fake_partial_annotation
% 
% 
%                 disp('ADDING TEST FOR PARTIAL ANNOTATION')
% 
% 
% 
% 
%                 single_info_trial.dataMat.Events(end+1).label = 'partial_annotation';
%                 single_info_trial.dataMat.Events(end).color = [0,1,0];
%                 single_info_trial.dataMat.Events(end).epochs = 1;
%                 single_info_trial.dataMat.Events(end).times = [sProcess.options.timewindow.Value{1}(1:2)]';
%                 single_info_trial.dataMat.Events(end).reactTimes = [];
%                 single_info_trial.dataMat.Events(end).select = 1;
%                 single_info_trial.dataMat.Events(end).notes = {[]};
% 
%                 if randomize_annotation_position_for_some_channels
% 
%                     % The usage for this is to randomly change the position of
%                     % 4 channels around each to random places around the head,
%                     % so the model does not learn the position
%                     % Besides the annotation on the events, the F matrix also
%                     % has to accommodate that change
% 
%                     nChannels = length(eyes_channel_Names)/2;  % The assumption here is that each cluster on each eye has the same number of channels
%                     clusters = ivadomed_getNchannelsAroundCenter(nChannels, single_info_trial.ChannelMat, figures_struct(iFile).FigureObject);
% 
%                     single_info_trial.dataMat.Events(end).channels = {[clusters.SelectedChannelsNames]};
%                     annotated_indicesOnChannelMat = [clusters.SelectedIndicesOnChannelMat];
% 
%                     eyes_channel_indicesOnChannelMat =  find(ismember({single_info_trial.ChannelMat.Channel.Name}, eyes_channel_Names));
% 
%     %                 hold on
%     %                 plot(clusters(1).CenterCoords(1), clusters(1).CenterCoords(2),'*b')
%     %                 plot(clusters(1).SelectedChannelsCoords(:,1),clusters(1).SelectedChannelsCoords(:,2),'bo')
%     %                 plot(clusters(2).CenterCoords(1), clusters(2).CenterCoords(2),'*r')
%     %                 plot(clusters(2).SelectedChannelsCoords(:,1),clusters(2).SelectedChannelsCoords(:,2),'ro')
%     %                 hold off
% 
%                 else  % Annotate only the 8 channels on each eye
%                     single_info_trial.dataMat.Events(end).channels = {eyes_channel_Names}; % Eyes
%                 end
% 
%             end
% 
% 
%             if randomize_annotation_position_for_some_channels && ~fake_partial_annotation
% 
%                 % The usage for this is to randomly change the position of
%                 % 4 channels around each to random places around the head,
%                 % so the model does not learn the position
%                 % Besides the annotation on the events, the F matrix also
%                 % has to accommodate that change
% 
%                 nChannels = 8;
%                 clusters = ivadomed_getNchannelsAroundCenter(nChannels, single_info_trial.ChannelMat, figures_struct(iFile).FigureObject);
%                 annotated_indicesOnChannelMat = [clusters.SelectedIndicesOnChannelMat];
%                 eyes_channel_indicesOnChannelMat =  find(ismember({single_info_trial.ChannelMat.Channel.Name}, eyes_channel_Names));
% 
%             else  % Annotate only 8 channels on each eye
%                 single_info_trial.dataMat.Events(end).channels = {eyes_channel_Names}; % Eyes
%             end
%         
%         end
%         
%         
%         
%         
        
        
    
    %% ADD A JITTER
    % We want to avoid the model learning the positioning of the event so
    % we crop the time dimension on both sides with a jitter
    
    if strcmp(sProcess.options.convert.Value, 'conversion') && sProcess.options.jitter.Value{1}~=0  % Only during training add a jitter - during (deep learning) segmentation the trial is not truncated
        current_Fs = round(1/diff(single_info_trial.dataMat.Time(1:2)));
        discardElementsBeginning = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);
        discardElementsEnd = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);

        single_info_trial.dataMat.Time = single_info_trial.dataMat.Time(1+discardElementsBeginning:end-discardElementsEnd);
        
        if randomize_annotation_position_for_some_channels
            temp = single_info_trial.dataMat.F(:,1+discardElementsBeginning:end-discardElementsEnd);
            temp_eyes = temp(eyes_channel_indicesOnChannelMat,:);
            temp_new_position_eyes = temp(annotated_indicesOnChannelMat,:);
            
            temp(eyes_channel_indicesOnChannelMat,:) = temp_new_position_eyes;
            temp(annotated_indicesOnChannelMat,:) = temp_eyes;
        	single_info_trial.dataMat.F = temp;
        else
            single_info_trial.dataMat.F = single_info_trial.dataMat.F(:,1+discardElementsBeginning:end-discardElementsEnd);
        end
    end
    
    
    %% CHANGE THE COLORMAP DISPLAYED ON THE BOTTOM LEFT
%     % Get colormap type
%     ColormapInfo = getappdata(GlobalData.DataSet.Figure.hFigure, 'Colormap');
%     
%     % ==== Update colormap ====
%     % Get colormap to use
%     sColormap = bst_colormaps('GetColormap', ColormapInfo.Type);
%     % Set figure colormap (for display of the colorbar only)
%     set(GlobalData.DataSet.Figure.hFigure, 'Colormap', sColormap.CMap);
%     set(GlobalData.DataSet.Figure.hFigure, 'Colormap', sColormap.CMap);
    

    %% Create NIFTIs for the selected modality (for MEG+EEG run it twice)
    modality = sProcess.options.modality.Comment(sProcess.options.modality.Value);
    if strcmp(modality{1}, 'MEG+EEG')
        modality = {'MEG', 'EEG'};  % The order matters, don't change until refactoring
    end
    
    for iModality = 1:length(modality)
    
        subject{iModality,1} = single_info_trial.subject;

        figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct, modality{iModality});

        % Select the appropriate sensors
        nElectrodes = 0;
        selectedChannels = [];

        for iChannel = 1:length(single_info_trial.ChannelMat.Channel)
           if strcmp(single_info_trial.ChannelMat.Channel(iChannel).Type, modality{iModality}) && ...
               single_info_trial.dataMat.ChannelFlag(iChannel)==1

                   %% TODO - ACCOMMODATE MORE HERE - fNIRS?
               nElectrodes = nElectrodes + 1;
               selectedChannels(end + 1) = iChannel;
           end
        end

        %% Drop out channels if requested
        if ~isempty(sProcess.options.channelDropOut.Value{1}) && sProcess.options.channelDropOut.Value{1}~=0
            nChannelsToDropout = randi(sProcess.options.channelDropOut.Value{1});
            iChannelsToDropout = selectedChannels(randi(length(selectedChannels), nChannelsToDropout,1));
            
            single_info_trial.dataMat.ChannelFlag(iChannelsToDropout) = -1;
            selectedChannels = selectedChannels(~ismember(selectedChannels,iChannelsToDropout));
        end
        
        %% Gather the topography slices to a single 3d matrix
        % Here the time dimension is the 3rd dimension
        [NIFTI, channels_pixel_coordinates] = channelMatrix2pixelMatrix(single_info_trial.dataMat.F, single_info_trial.dataMat.Time, single_info_trial.ChannelMat, selectedChannels, iFile, figures_struct, 0, sProcess);
    %     figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality{iModality});

        %% Get the output filename

        % Substitute the voxels with the 2D slices created from the 2dlayout
        % topography
        single_info_trial.sMri.Cube = NIFTI;

        %% Export the created cube to NIFTI
        OutputMriFile{iModality,1} = export2NIFTI(single_info_trial.sMri, single_info_trial.parentPath, single_info_trial.OutputMriFile{iModality});

        %% Export times if doing training, and channels' coordinates if doing segmentation
        export_Channels_Times(single_info_trial, channels_pixel_coordinates, single_info_trial.dataMat.Time', sProcess.options.convert.Value);


        %% Create derivative

        % The derivative will be based on a period of time that is annotated to
        % be the Ground truth.

        % First check that an event label was selected by the user for
        % annotation
        % In the case of extended event, only that period of time will be annotated
        % In the case of a simple event, the annotation will be converted to
        % extended based on the time-window values selected by the user,
        % RELATIVE TO THE SIMPLE EVENT OCCURENCE e.g. [-50,50] ms around the
        % event.
        % In no eventlabel is selected, the annotation will be based on the time-window
        % selected, with the TIMING MATCHING THE TRIAL-TIME VALUES

        if strcmp(sProcess.options.convert.Value, 'conversion')  % Create the derivative Only during conversion for training

            F_derivative = zeros(size(single_info_trial.dataMat.F));    

            if isempty(single_info_trial.dataMat.Events)
                iAllSelectedEvents = [];
            else
                iAllSelectedEvents = find(ismember({single_info_trial.dataMat.Events.label}, strsplit(sProcess.options.eventname.Value,{',',' '})));
            end
            annotationValue = 0;

            % Make a distinction between trials that will be used as baselines
            % (no-annotation - we are just keeping a black NIFTI)
            if isempty(strfind(single_info_trial.trial{iModality}, 'baseline'))

                if ~isempty(iAllSelectedEvents)  % Selected event
                    for iSelectedEvent = iAllSelectedEvents
                        annotationValue = annotationValue+1;
                        isExtended = size(single_info_trial.dataMat.Events(iSelectedEvent).times,1)>1;                                       

                        if isExtended
                            % EXTENDED EVENTS - ANNOTATE BASED ON THEM ONLY
                            for iEvent = 1:size(single_info_trial.dataMat.Events(iSelectedEvent).times,2)
                                iAnnotation_time_edges  = bst_closest(single_info_trial.dataMat.Events(iSelectedEvent).times(:,iEvent)', single_info_trial.dataMat.Time);

                                % If no specific channels are annotated, annotate the entire slice
                                if isempty(single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent})
                                    if sProcess.options.gaussian_annot.Value
                                        F_derivative = gaussian_annotation(F_derivative, [], iAnnotation_time_edges);
                                    else                                    
                                        F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                    end
                                else
                                    iAnnotation_channels  = find(ismember({single_info_trial.ChannelMat.Channel.Name}, single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                                    selectedChannels = iAnnotation_channels;
                                    
                                    if sProcess.options.gaussian_annot.Value
                                        F_derivative = gaussian_annotation(F_derivative, iAnnotation_channels, iAnnotation_time_edges);
                                    else                                    
                                        F_derivative(iAnnotation_channels,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                    end
                                end
                            end
                        else
                            % SIMPLE EVENTS - ANNOTATE BASED ON THE TIME WINDOW
                            % SELECTED AROUND THEM
                            for iEvent = 1:size(single_info_trial.dataMat.Events(iSelectedEvent).times,2)
                                % Here the annotation is defined by the selected event
                                % and the time-window selected around it
                                iAnnotation_time_edges  = bst_closest(single_info_trial.dataMat.Events(iSelectedEvent).times(iEvent)+sProcess.options.timewindow.Value{1}, single_info_trial.dataMat.Time);

                                % If no specific channels are annotated, annotate the entire slice
                                if isempty(single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent})
                                    if sProcess.options.gaussian_annot.Value
                                        F_derivative = gaussian_annotation(F_derivative, [], iAnnotation_time_edges);
                                    else                                    
                                        F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                    end
                                else
                                    iAnnotation_channels  = find(ismember({single_info_trial.ChannelMat.Channel.Name}, single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                                    selectedChannels = iAnnotation_channels;
                                    
                                    if sProcess.options.gaussian_annot.Value
                                        F_derivative = gaussian_annotation(F_derivative, iAnnotation_channels, iAnnotation_time_edges);
                                    else                                    
                                        F_derivative(iAnnotation_channels,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                    end
                                end
                            end
                        end
                    end
                else  % No event selected - ANNOTATE BASED ON THE SELECTED TIME WINDOW WITHIN THE TIME IN TRIAL
                    annotationValue = annotationValue+1;
                    iAnnotation_time_edges  = bst_closest(sProcess.options.timewindow.Value{1}, single_info_trial.dataMat.Time);
                    % Annotate the entire slice
                    F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                end

            else
                disp('Baseline trial detected - No annotation on its derivative')

            end

            figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct, modality{iModality});
            [NIFTI_derivative, channels_pixel_coordinates] = channelMatrix2pixelMatrix(F_derivative, single_info_trial.dataMat.Time, single_info_trial.ChannelMat, selectedChannels, iFile, figures_struct, 1, sProcess);
            figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality{iModality});


            % Set the values to 0 and 1 for the annotations
            % Hard threshold
            NIFTI_derivative = double(NIFTI_derivative)/max(max(max(double(NIFTI_derivative))));

            
            % Apply the soft annotation threshold only if the gaussian
            % annotation is not selected
            if ~sProcess.options.gaussian_annot.Value
                % If no value is selected for soft thesholding the annotation, apply a
                % hard annotation at 0.5
                if isempty(sProcess.options.annotthresh.Value{1})
                    NIFTI_derivative(NIFTI_derivative<0.5) = 0;
                    NIFTI_derivative(NIFTI_derivative>=0.5) = 1;
                else
                    % In case of soft-annotation thresholding, assign everything below
                    % the threshold to 0 - BUT KEEP THE VALUES AS THEY ARE ABOVE THE
                    % THRESHOLD
                    NIFTI_derivative(NIFTI_derivative<sProcess.options.annotthresh.Value{1}) = 0;
                end
            end

            % Annotate derivative
            single_info_trial.sMri.Cube = NIFTI_derivative;

            %% Export the created cube to NIFTI
            export2NIFTI(single_info_trial.sMri, single_info_trial.parentPath, single_info_trial.OutputMriFileDerivative{iModality});

        else
            figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality{iModality});
        end
    end
end

function figures_struct = open_close_topography_window(FileName, action, iFile, figures_struct, Modality)
    global GlobalData
    if strcmp(action, 'open')
        %% Open a window to inherit properties
        %[hFig, iDS, iFig] = view_topography(DataFile, Modality, TopoType, F)
        % TODO - consider adding flag on view_topography for not displaying the
        % figure when it is for Ivadomed
        % Modality       : {'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'ECOG', 'SEEG', 'NIRS'}
        % TopoType       : {'3DSensorCap', '2DDisc', '2DSensorCap', 2DLayout', '3DElectrodes', '3DElectrodes-Cortex', '3DElectrodes-Head', '3DElectrodes-MRI', '3DOptodes', '2DElectrodes'}
        
            
        [hFig, iDS, iFig] = view_topography(FileName, Modality, '2DSensorCap');     

%         hFig.CurrentAxes.PlotBoxAspectRatio = [1,1,1];

%         set(hFig, 'Visible', 'off');
%         set(hFig, 'Position', [hFig.Position(1) hFig.Position(2) 355 258]);  % THE AXIS IS [~,~,277.5, 238] WITH THIS CONFIGURATION
        set(hFig, 'Resize', 0);
        set(hFig, 'Position', [hFig.Position(1) hFig.Position(2) 177.5 129]);
        
%         AxesH = hFig.Children(2);
        AxesH.PlotBoxAspectRatio = [1,1,1];
        pause(.05);
%         set(AxesH, 'Units', 'pixels', 'Position', [10, 10, 100, 86]);
%         
%                 daspect([1 1 1])

        
        
        % Find index that just opened figure corresponds to (this is done for enabling parallelization)
        all_datafiles = {GlobalData.DataSet.DataFile};
        [temp, index] = ismember(FileName, all_datafiles);
        
        figures_struct(iFile).FigureObject = GlobalData.DataSet(index).Figure;
        figures_struct(iFile).Status       = 'Open';
        figures_struct(iFile).Modality     = Modality;
        

        %TODO - GET COLORMAP TYPE AUTOMATICALLY (meg - eeg - fnirs)
        % Get figure colormap
        
        ColormapType = lower(Modality);  % meg, eeg, fnirs
        colormapName = 'gray';
%         colormapName = 'mine';
        bst_colormaps('SetColormapName', ColormapType, colormapName);


        % If there are any contours, remove them
        % Delete contour objects
% % %         delete(TopoHandles.hContours);
% % %         GlobalData.DataSet(iDS).Figure(iFig).Handles.hContours = [];

    elseif strcmp(action, 'close')
        % Close window
%         close(GlobalData.DataSet(iFile).Figure.hFigure)
        close(figures_struct(iFile).FigureObject.hFigure)
        figures_struct(iFile).Status = 'Closed';
    end
end


function [NIFTI, channels_pixel_coordinates] = channelMatrix2pixelMatrix(F, Time, ChannelMat, selectedChannels, iFile, figures_struct, isDerivative, sProcess)
    % global GlobalData

%      %  %  %  TODO - CONSIDER MAKING ALL VALUES POSITIVE AND CHANGE THE
%         %  MIN MAX TO [0, MIN+MAX]
% THIS IS STILL NOT WORKING - check for usage with int8 or uint8
%     F = abs(min(min(F(selectedChannels,:)))) + F;
    

    % ATTEMPT TO PARALLELIZE FIGURES
% % %     % Create a new figure that will have a copy of the objects from the
% % %     % original
% % %     hFig2 = figure('visible','off');
% % %     set(hFig2, 'units', 'pixels');
% % %     set(hFig2, 'Position', [0 0 710 556]);  % Default values of single 2dlayout figure
% % %     ax2 = copyobj(figures_struct(1).FigureObject.hFigure.CurrentAxes,hFig2);
% % % %     ax1Chil = figures_struct(1).FigureObject.hFigure.CurrentAxes.Children; 
% % % %     copyobj(ax1Chil, ax2)
% % % %     set(ax2.Children(1), 'FaceVertexCData', DataToPlot, 'EdgeColor', 'none');
    
    
    % GLOBAL MIN_MAX FOR EACH TRIAL
    the_min = min(min(F(selectedChannels,:)));
    the_max = max(max(F(selectedChannels,:)));
    
    % This is altering the EEG 2D display - NOT THE COLORBAR ON THE BOTTOM
    % RIGHT - THE COLORBAR NEEDS TO BE ADDRESSED
%     GlobalData.DataSet(iFile).Figure.Handles.DataMinMax = [the_min, the_max];
    figures_struct(iFile).FigureObject.Handles.DataMinMax = [the_min, the_max];    
    caxis([the_min the_max]);

            
    delete(figures_struct(iFile).FigureObject.hFigure.Children(1)) % Gets rid of the colorbar object

    img = getframe(figures_struct(iFile).FigureObject.hFigure.Children);
    [height,width,~] = size(img.cdata);

        
    NIFTI = zeros(height, width, length(Time), 'uint8');
    for iTime = 1:length(Time)
        
        if ~isDerivative
            DataToPlot = F(selectedChannels,iTime);
            iChannel = selectedChannels;
        else
            [tmp,I,J] = intersect(selectedChannels, figures_struct(iFile).FigureObject.SelectedChannels);
            iChannel = J';
%             plot(figures_struct(iFile).FigureObject.Handles.MarkersLocs(iChannel,1), figures_struct(iFile).FigureObject.Handles.MarkersLocs(iChannel,2),'g.')
%             for i = 1:length(iChannel)
%                 text(figures_struct(iFile).FigureObject.Handles.MarkersLocs(iChannel(i),1), figures_struct(iFile).FigureObject.Handles.MarkersLocs(iChannel(i),2),ChannelMat.Channel(selectedChannels(i)).Name)
%             end            
            DataToPlot = F(selectedChannels,iTime);
        end

        % ===== APPLY TRANSFORMATION =====
        % Mapping on a different surface (magnetic source reconstruction of just smooth display)
%         if ~isempty(GlobalData.DataSet(iFile).Figure.Handles.Wmat)
        if ~isempty(figures_struct(iFile).FigureObject.Handles.Wmat)
            % Apply interpolation matrix sensors => display surface
%             if (size(GlobalData.DataSet(iFile).Figure.Handles.Wmat,1) == length(DataToPlot))
            if (size(figures_struct(iFile).FigureObject.Handles.Wmat,2) == length(DataToPlot)) % Topomap checks for the other dimension, it might be wrong
%                 DataToPlot = full(GlobalData.DataSet(iFile).Figure.Handles.Wmat * DataToPlot);
                DataToPlot = full(figures_struct(iFile).FigureObject.Handles.Wmat * DataToPlot);
            % Find first corresponding indices
            else
%                 [tmp,I,J] = intersect(selectedChannels, GlobalData.DataSet(iDS).Figure(iFig).SelectedChannels);
%                 [tmp,I,J] = intersect(selectedChannels, GlobalData.DataSet(iFile).Figure.SelectedChannels);
                [tmp,I,J] = intersect(selectedChannels, figures_struct(iFile).FigureObject.SelectedChannels);
%                 DataToPlot = full(GlobalData.DataSet(iFile).Figure.Handles.Wmat(:,J) * DataToPlot(I));
                DataToPlot = full(figures_struct(iFile).FigureObject.Handles.Wmat(:,J) * DataToPlot(I));
            end
        end         


%         set(GlobalData.DataSet(iFile).Figure.Handles.hSurf, 'FaceVertexCData', DataToPlot, 'EdgeColor', 'none');
        set(figures_struct(iFile).FigureObject.Handles.hSurf, 'FaceVertexCData', DataToPlot, 'EdgeColor', 'none');

        % Check exporting image
        img = getframe(figures_struct(iFile).FigureObject.hFigure.Children);
        img_gray= rgb2gray(img.cdata);
        
        if isDerivative && all(DataToPlot==0) % This is done since even if all channels are 0, there is still a gray image of the topography displayed to distringuish from the background
            img_gray(img_gray<170)=0;
        elseif isDerivative
            
            if ~sProcess.options.gaussian_annot.Value
                threshold = 145;  % For annotating a single channel assigned it to 133
                img_gray(img_gray<threshold)=0;
                img_gray(img_gray>=threshold)=1;
            end
        end
        
        NIFTI(:,:,iTime) = img_gray;
        
    end
    
    
    %% Change dimensions to fit the NIFTI requirements - FSLeyes displays the slices with the correct orientation after the flip
    NIFTI = flip(permute(NIFTI,[2,1,3]),2);
    
    %% Get electrode position in pixel coordinates - TODO - Still hardcoded
    axes_handle = figures_struct(iFile).FigureObject.hFigure.Children;
    set(axes_handle,'units','pixels');
    pos = get(axes_handle,'position');
    xlim = get(axes_handle,'xlim');
    ylim = get(axes_handle,'ylim');
    
    % Get the monitor resolution
    set(0,'units','pixels')  
    %Obtains this pixel information
    Pix_SS = get(0,'screensize');

    if all(Pix_SS == [1 1 2560 1440])
        if strcmp(figures_struct(iFile).Modality, 'MEG')
            % MONITOR
            % HARDCODED CHANNEL POSITION CORRECTION - TODO
            y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
            y_in_pixels = y_in_pixels/0.84;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
            x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
            x_in_pixels = 2 + x_in_pixels/1.25;
        elseif strcmp(figures_struct(iFile).Modality, 'EEG')
            % MONITOR
            % HARDCODED CHANNEL POSITION CORRECTION - TODO
            y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
            y_in_pixels = 2 + y_in_pixels/0.86;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
            x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
            x_in_pixels = 2 + x_in_pixels/1.25;
        end
    elseif all(Pix_SS == [1 1 1920 1080])
        if strcmp(figures_struct(iFile).Modality, 'MEG')
            % LAPTOP
            % HARDCODED CHANNEL POSITION CORRECTION - TODO
            y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
            y_in_pixels = 0 + y_in_pixels/0.836;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
            x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
            x_in_pixels = 1 + x_in_pixels/1.23;
        elseif strcmp(figures_struct(iFile).Modality, 'EEG')
            % LAPTOP
            % HARDCODED CHANNEL POSITION CORRECTION - TODO
            y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
            y_in_pixels = 1 + y_in_pixels/0.92;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
            x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
            x_in_pixels = 2 + x_in_pixels/1.15;
        end
    elseif all(Pix_SS == [1 1 3840 2160])
        if strcmp(figures_struct(iFile).Modality, 'MEG')
            % LAPTOP
            % HARDCODED CHANNEL POSITION CORRECTION - TODO
            y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
            y_in_pixels = 2.4 + y_in_pixels/0.87;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
            x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
            x_in_pixels = 2 + x_in_pixels/2.32;
        elseif strcmp(figures_struct(iFile).Modality, 'EEG')
            % LAPTOP
            % HARDCODED CHANNEL POSITION CORRECTION - TODO
            y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
            y_in_pixels = 3 + y_in_pixels/0.89;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
            x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
            x_in_pixels = 3 + x_in_pixels/2.4;
        end
    else
        error('Unknown monitor - Need to calibrate - Also time to get rid of these harcoded parts!')
    end

    
    
    %% Gather channel coordinates in a struct
    channels_pixel_coordinates = struct;

    for i = 1:length(selectedChannels)
        channels_pixel_coordinates(i).ChannelNames = [ChannelMat.Channel(selectedChannels(i)).Name];
        channels_pixel_coordinates(i).x_coordinates = round(x_in_pixels(i));
        channels_pixel_coordinates(i).y_coordinates = round(y_in_pixels(i));
    end
    
    %%
%      disp(1)
%     h = figure(10);
%     imagesc(squeeze(NIFTI(:,:,75)))
%     colormap('gray')
% 
%     hold on
%     plot(y_in_pixels, x_in_pixels,'*r')
%     hold off

    
end


function OutputMriFile_full = export2NIFTI(sMri, parentPath, OutputMriFile)
    %% Export to NIFTI

    OutputMriFile_full = bst_fullfile(parentPath, OutputMriFile);
    
    % Create the output folder first
    [filepath,name,ext] = fileparts(OutputMriFile_full);
    if ~exist(filepath, 'dir')  % This avoids a warning if the folder already exists
        mkdir(bst_fileparts(OutputMriFile_full))
    end    

    % Export (.nii)
    out_mri_nii(sMri, OutputMriFile_full);
    
    % Zip nifti (create .nii.gz)
    gzip(OutputMriFile_full)
    % Delete nifti (.nii)
    delete(OutputMriFile_full)
    
    OutputMriFile_full = [OutputMriFile_full '.gz'];
    
end


function export_Channels_Times(single_info_trial, channels_pixel_coordinates, Time, method)

    switch method
        case 'conversion' 
            % For segmentation we dont really need this - we have the info from the
            % trials themselved - they are not truncated as is the case with the
            % training part
            % Create the output folder first
            OutputTimesFile = bst_fullfile(single_info_trial.channels_times_path, single_info_trial.OutputTimesFile);
            
            [filepath,name,ext] = fileparts(OutputTimesFile);
            if ~exist(filepath, 'dir')  % This avoids a warning if the folder already exists
                mkdir(bst_fileparts(OutputTimesFile))
            end
            
            % Export times to a .csv (This is needed since the trials have been truncated from the JITTER parameter)
            writematrix(Time, OutputTimesFile)
            
        case 'segmentation'
            % Create the output folder first
            OutputChannelsFile = bst_fullfile(single_info_trial.channels_times_path, single_info_trial.OutputChannelsFile);
            [filepath,name,ext] = fileparts(OutputChannelsFile);
            if ~exist(filepath, 'dir')  % This avoids a warning if the folder already exists
                mkdir(bst_fileparts(OutputChannelsFile))
            end    

            % Export the channel coordinates to a .csv file
            writetable(struct2table(channels_pixel_coordinates), OutputChannelsFile, 'Delimiter', '\t')
    end
end


function export_participants_tsv(parentPath, subjects)
    if ~exist(parentPath, 'dir')  % This avoids a warning if the folder already exists
        mkdir(parentPath)
    end    
    
    participants_data = struct;
    
    for i = 1:length(subjects)
        participants_data(i).participant_id = ['sub-' subjects{i}];
        participants_data(i).sex = 'na';
        participants_data(i).age = 'na';
    end
        
    % Writetable didn't allow export in .tsv - I rename it after
    writetable(struct2table(participants_data), bst_fullfile(parentPath, 'participants.txt'), 'Delimiter', '\t')
    movefile(bst_fullfile(parentPath, 'participants.txt'), bst_fullfile(parentPath, 'participants.tsv'))
end


function export_participants_json(parentPath)
    text = '{\n"participant_id": {\n\t"Description": "Unique ID",\n\t"LongName": "Participant ID"\n\t},\n"sex": {\n\t"Description": "M or F",\n\t"LongName": "Participant sex"\n\t},\n"age": {\n\t"Description": "yy",\n\t"LongName": "Participant age"\n\t}\n}';

    fileID = fopen(bst_fullfile(parentPath, 'participants.json'),'w');
    fprintf(fileID,text);
    fclose(fileID);
end


function export_dataset_description(parentPath)
    text = '{\n\t"BIDSVersion": "1.6.0",\n\t"Name": "Ivadomed@Brainstorm"\n}';
    
    fileID = fopen(bst_fullfile(parentPath, 'dataset_description.json'),'w');
    fprintf(fileID,text);
    fclose(fileID);
end


function export_readme(parentPath)
    text = 'Converted BIDS dataset from Brainstorm trials';
    
    fileID = fopen(bst_fullfile(parentPath, 'README'),'w');
    fprintf(fileID,text);
    fclose(fileID);
end


function modify_config_json(parentPath, modality, annotation, contrast_params_txt, sProcess, sInputs_filenames)


    %% CHANGE THE CONFIG FILE TO RUN LOCALLY
    % Grab the config.json file that was used and assign the gpu that the
    % user selected
    configFile = bst_fullfile(bst_get('BrainstormHomeDir'), 'external', 'ivadomed', 'config_epilepsy_template.json');
    
    fid = fopen(configFile);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);          
    
    %Substitute null values with nan - This is needed cause jsonencode
    %changes null values to [] and ultimately ivadomed throws errors
    str = strrep(str, 'null', 'NaN');
    
    config_struct = jsondecode(str);
    
    [temp1,folder_output,temp2] = bst_fileparts(parentPath);

    % Change input-output folders
    config_struct.loader_parameters.path_data = ['/home/GRAMES.POLYMTL.CA/u111358/data_nvme_u111358/EEG-ivado/epilepsy/distant_spikes_2_seconds_gaussian_annotation/data_' folder_output '_gaussian_annotation'];
    config_struct.path_output = ['/home/GRAMES.POLYMTL.CA/u111358/data_nvme_u111358/EEG-ivado/epilepsy/distant_spikes_2_seconds_gaussian_annotation/output_' folder_output '_gaussian_annotation'];
    
    % Change the contrast_params{training_validation, testing}
    contrast_params_cell = cellfun(@(x) strrep(x,'"',''),contrast_params_txt, 'UniformOutput',false);
    config_struct.loader_parameters.contrast_params.training_validation = contrast_params_cell;
    config_struct.loader_parameters.contrast_params.testing = contrast_params_cell;
    
    config_struct.model_name = [modality '_model'];
    config_struct.loader_parameters.target_suffix = {['_' annotation]};
    
    config_struct.gpu_ids = {0};
    
    % Add the Brainstorm parameters on the json file
    config_struct.brainstorm = struct;
    config_struct.brainstorm.modality = sProcess.options.modality.Comment{sProcess.options.modality.Value};
    config_struct.brainstorm.event_for_ground_truth = sProcess.options.eventname.Value;
    config_struct.brainstorm.channel_drop_out = sProcess.options.channelDropOut.Value{1};
    config_struct.brainstorm.annotations_time_window = sProcess.options.timewindow.Value{1};
    config_struct.brainstorm.fs = sProcess.options.fs.Value{1};
    config_struct.brainstorm.jitter = sProcess.options.jitter.Value{1};
    config_struct.brainstorm.soft_annotation_threshold = sProcess.options.annotthresh.Value{1};
    config_struct.brainstorm.bids_folder_creation_mode = sProcess.options.bidsFolders.Comment{sProcess.options.bidsFolders.Value};
    
    % Adding an index to help identify the trial
    for i = 1:length(sInputs_filenames)
        sInputs_filenames{i} = [num2str(i) ': ' sInputs_filenames{i}];
    end
    config_struct.brainstorm.sInputs = sInputs_filenames;
    
    % Save back to json
    txt = jsonencode(config_struct, 'PrettyPrint', true);
    
    new_configFile = bst_fullfile(parentPath, 'config_for_training.json');
    fid = fopen(new_configFile, 'w');
    fwrite(fid, txt);
    fclose(fid);
    
end



function F_derivative = gaussian_annotation(F_derivative, iAnnotation_channels, iAnnotation_time_edges)
    % This function returns the derivatives with a gaussian annotation.
    % instead of a hard 0,1 annotation.
    % Gaussian annotations can be used for soft training
    % The config file in IVADOMED needs to be modified to make use of these
    % annotations!

    N = iAnnotation_time_edges(2)-iAnnotation_time_edges(1)+1;

    w = gausswin(N,3)';
    
    single_channel_gaussian_annotation = zeros(1, size(F_derivative,2));
    single_channel_gaussian_annotation(iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = w;
    
    
    % Now insert it on a signal 
    if isempty(iAnnotation_channels)
        F_derivative = repmat(single_channel_gaussian_annotation, size(F_derivative,1), 1);
    else
        F_derivative = repmat(single_channel_gaussian_annotation, length(iAnnotation_channels), 1);
    end

end

