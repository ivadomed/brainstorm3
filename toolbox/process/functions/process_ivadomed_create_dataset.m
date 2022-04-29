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
% Author: Konstantinos Nasiotis, 2021-2022

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
    
    % Parent Folder to store the dataset in
    SelectOptions = {...
        '', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'Folder to store dataset in...', ...     % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                    % Selection mode: {single,multiple}
        'dirs', ...                        % Selection mode: {files,dirs,files_and_dirs}
        {{'.folder'}, 'BIDS dataset folder', 'IVADOMED'}, ... % Available file formats
        []};                               % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    
    sProcess.options.label.Comment = '<B>BIDS folder</B>';
    sProcess.options.label.Type    = 'label';
    % Option: Dataset folder
    sProcess.options.BIDSfolder.Comment = 'Parent folder to store BIDS dataset in:';
    sProcess.options.BIDSfolder.Type    = 'filename';
    sProcess.options.BIDSfolder.Value   = SelectOptions;
    
    % Dataset folder help comment
    sProcess.options.BIDSfolder_help.Comment = '<I><FONT color="#777777">Leave empty to store in Brainstorm temp folder</FONT></I>';
    sProcess.options.BIDSfolder_help.Type    = 'label';
    
    % Modality Selection
    sProcess.options.label11.Comment = '<BR><B>Modality selection:</B>';
    sProcess.options.label11.Type    = 'label';
    sProcess.options.modality.Comment = {'MEG', 'EEG', 'MEG+EEG', 'fNIRS'};
    sProcess.options.modality.Type    = 'radio';
    sProcess.options.modality.Value   = 1;
    
    % Annotation parameters
    sProcess.options.label10.Comment = '<BR><B>Annotation parameters:</B>';
    sProcess.options.label10.Type    = 'label';
    % Event name
    sProcess.options.eventname.Comment = 'Events for ground truth (separate with commas, spaces or semicolons)';
    sProcess.options.eventname.Type    = 'text';
    sProcess.options.eventname.Value   = 'event1';
    % Options: Segment around spike
    sProcess.options.timewindow_annot.Comment  = 'Annotations Time window: ';
    sProcess.options.timewindow_annot.Type     = 'range';
    sProcess.options.timewindow_annot.Value    = {[-0.150, 0.150],'ms',[]};
    % Event help comment
    sProcess.options.timewindow_annot_help.Comment = '<I><FONT color="#777777">This time window is only used for annotating around single events</FONT></I>';
    sProcess.options.timewindow_annot_help.Type    = 'label';
    % Method: BIDS annotation type selection
    sProcess.options.whole_partial_annotationLabel.Comment = '<I><FONT color="#FF0000">Whole Head annotation (all channels) or partial (annotate specific channels)</FONT></I>';
    sProcess.options.whole_partial_annotationLabel.Type    = 'label';
    sProcess.options.whole_partial_annotation.Comment = {'Whole', 'Partial'};
    sProcess.options.whole_partial_annotation.Type    = 'radio';
    sProcess.options.whole_partial_annotation.Value   = 1;
    
    % Use gaussian soft annotation
    sProcess.options.gaussian_annot.Comment = 'Gaussian annotation';
    sProcess.options.gaussian_annot.Type    = 'checkbox';
    sProcess.options.gaussian_annot.Value   = 0;
    sProcess.options.gaussian_annot_help.Comment = '<I><FONT color="#777777">The annotation within the annotation window will be a gaussian function</FONT></I>';
    sProcess.options.gaussian_annot_help.Type = 'label';
    
    % Trials selection
    sProcess.options.label4.Comment = '<BR><B>Trials selection:</B>';
    sProcess.options.label4.Type    = 'label';
    % Number of IED segments selected
    sProcess.options.segment_number.Comment = 'Resampling rate <I><FONT color="#777777">(empty for no resampling)</FONT></I>';
    sProcess.options.segment_number.Type    = 'value';
    sProcess.options.segment_number.Value   = {100, 'segments', 0};
    % Include trials without annotations to the dataset?
    sProcess.options.baselines.Comment = 'Include trials that dont have selected event(s) <I><FONT color="#777777"> (Annotation NIFTIs are zeroed)</FONT></I>';
    sProcess.options.baselines.Type    = 'checkbox';
    sProcess.options.baselines.Value   = 0;
    
    % BIDS
    sProcess.options.label1.Comment = '<BR><B>BIDS conversion parameters:</B>';
    sProcess.options.label1.Type    = 'label';
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
    
    
    % Method: BIDS subject selection
    sProcess.options.label2.Comment = '<BR><B>BIDS folders creation </B>';
    sProcess.options.label2.Type    = 'label';
    sProcess.options.bidsFolders.Comment = {'Normal', 'Separate runs/sessions as different subjects', 'Separate each trial as different subjects'};
    sProcess.options.bidsFolders.Type    = 'radio';
    sProcess.options.bidsFolders.Value   = 1;
    
    % Conversion of both trials and their derivatives
    sProcess.options.convert.Type   = 'text';
    sProcess.options.convert.Value  = 'conversion';  % Other option: 'segmentation'
    sProcess.options.convert.Hidden = 1;
    
    % Display example image in FSLeyes
    sProcess.options.label3.Comment = '<BR><B>FSLeyes </B>';
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


    %% Close all existing figures
    bst_memory('UnloadAll', 'Forced');

    
    %% Sampling rate to be used when creating the NIFTI files
    wanted_Fs = sProcess.options.fs.Value{1};
    
    %% Modality selected
    modality = sProcess.options.modality.Comment{sProcess.options.modality.Value};
    
    
    %% Do some checks on the parameters
    
    if isempty(sProcess.options.eventname.Value)
        error('No event label has been selected to be used as ground truth')
    end
    
    subjects_inputs = string([]);
    for iInput = 1:length(sInputs)
        subjects_inputs(iInput) = lower(str_remove_spec_chars(sInputs(iInput).SubjectName));
    end
    
    if strcmp(sProcess.options.convert.Value, 'conversion')
    
        % In case the selected event is single event, or the eventname isempty,
        % make sure the time-window has values
        inputs_to_remove = false(length(sInputs),1);

        for iInput = 1:length(sInputs)
            dataMat = in_bst(sInputs(iInput).FileName, 'Events');
            events = dataMat.Events;

            Time = in_bst(sInputs(iInput).FileName, 'Time');
            
            if isfield(events, 'label')
                [areSelectedEventsPresent, indices] = ismember(strsplit(sProcess.options.eventname.Value, {' ', ',', ';'}), {events.label});
            else
                areSelectedEventsPresent = false;
            end

            if ~any(areSelectedEventsPresent)
                inputs_to_remove(iInput) = true;
                bst_report('Warning', sProcess, sInputs(1), ['The selected events do not exist within trial: ' dataMat.Comment ' . Ignoring this trial']);
            else
%                     % TODO - THIS IS WRONG - IT SHOULD BE REJECTED IF ALL ANNOTATIONS ARE OUT OF BOUNDS, NOT JUST THE FIRST AND LAST
%                     if Time(1) > sProcess.options.timewindow_annot.Value{1}(1) + events(indices).times(1) || Time(end) < events(indices).times(end) + sProcess.options.timewindow_annot.Value{1}(2)
%                         inputs_to_remove(iInput) = true;
%                         bst_report('Warning', sProcess, sInputs, ['The time window selected for annotation is not within the Time segment of trial: ' dataMat.Comment ' . Ignoring this trial']);                    
%                     end
                continue

            end
        end
        
        subjects_name = unique(subjects_inputs);
        for isub = 1:length(subjects_name)
            
            sub_inputs = find(subjects_inputs == subjects_name(isub));
            inputs_to_remove_sub = inputs_to_remove(sub_inputs);
            input_event = find(inputs_to_remove_sub == false);
            input_no_event = find(inputs_to_remove_sub == true);
            sub_inputs_event = sub_inputs(input_event);
            sub_inputs_no_event = sub_inputs(input_no_event);
            n_event = length(input_event);
       
            if n_event > sProcess.options.segment_number.Value{1}
                randomIndex_to_remove = randperm(n_event, n_event - sProcess.options.segment_number.Value{1});
                inputs_to_remove(sub_inputs_event(randomIndex_to_remove)) = true;
            end
            
            if sProcess.options.baselines.Value
                n_no_event = length(input_no_event);
                n_no_event_to_add = min(n_event, sProcess.options.segment_number.Value{1});
                randomIndex_to_add = randperm(n_no_event, n_no_event_to_add);
                inputs_to_remove(sub_inputs_no_event(randomIndex_to_add)) = false;
            end
                
        end
        
%         
        % The following works on Matlab R2021a - I think older versions
        % need another command to squeeze the empty structure entries - TODO
        sInputs(inputs_to_remove) = [];
    end
    
    
    if isempty(sInputs)
        bst_report('Error', sProcess, sInputs, 'No inputs selected');
        OutputFiles = {};
        return
    end
    %% Gather filenames for all files here - This shouldn't create a memory issue
    % Reason why all of them are imported here is that in_bst doesn't like
    % to be parallelized (as far as I checked), so can't call it within 
    % convertTopography2matrix
    
    % Organize the folder hierarchy
    protocol = bst_get('ProtocolInfo');
    if strcmp(sProcess.options.convert.Value, 'conversion')
        
        % If the user hasn't selected a parent folder for the BIDS dataset,
        % save everything in the temp folder.
        % ATTENTION - The temp folder is emptied after each call of the
        % function
        if isempty(sProcess.options.BIDSfolder.Value{1})
            
            parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                           'IvadomedNiftiFiles', ...
                           [protocol.Comment '_' modality '_dataset']);
        else
            % In case the user has selected a parent folder to save the
            % BIDS folder, this snippet will create a sequential index for
            % easy identification
            
            % List the folders that exist within the parent folder selected
            files = dir(sProcess.options.BIDSfolder.Value{1});
            iDirs = find([files.isdir]);
            
            maxID = 0;
            for iFolder = iDirs
                if strfind(files(iFolder).name, [protocol.Comment '_' modality '_datasetID_'])
                    
                    separate = strsplit(files(iFolder).name,'_');
                    ID = str2double(separate{end});
                    
                    if ID>maxID
                        maxID = ID;
                        separate{end} = num2str(ID+1);
                        folder_name = join(separate, '_');
                        parentPath = bst_fullfile(files(iFolder).folder, folder_name{1});
                    end
                end
            end
            
            % In case there is no folder with datasetID, create the first
            % one
            if maxID==0
                parentPath = bst_fullfile(sProcess.options.BIDSfolder.Value{1}, [protocol.Comment '_' modality '_datasetID_1']);
            end
        end
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
                   
    % Differentiate simple conversion and segmentation paths
    if strcmp(sProcess.options.convert.Value, 'segmentation')
        channels_times_path = [channels_times_path '-segmentation'];
        parentPath = [parentPath '-segmentation'];
    end
    
        
    annotation = str_remove_spec_chars(sProcess.options.eventname.Value);
                   
    % Hack to accommodate ivadomed derivative selection:
    % https://github.com/ivadomed/ivadomed/blob/master/ivadomed/loader/utils.py # L812
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'; % This hack accommodates up to 240 trials within a run - for more find another solution 
                                            % - like double letters (not the same though or the same IVADOMED loader problem would occur)
                 
    
    summed_trial_duration = 0; % This needs to be included in the config file, since the model needs to be applied on trials with the same timelength
    
    info_trials = struct;
    
    txt  = [];
    
    for iInput = 1:length(sInputs)
        info_trials(iInput).FileName = sInputs(iInput).FileName;
        info_trials(iInput).subject = lower(str_remove_spec_chars(sInputs(iInput).SubjectName));
        info_trials(iInput).session = lower(str_remove_spec_chars(sInputs(iInput).Condition));
        subjects_inputs(iInput) = lower(str_remove_spec_chars(sInputs(iInput).SubjectName));
        % The trial # needs special attention
        splitComment = split(sInputs(iInput).Comment,{'(#',')'});
        comment = lower(str_remove_spec_chars(splitComment{1}));
        iEpoch = str2double(splitComment{2});
        
        if ~strcmp(modality, 'MEG+EEG')
            if ~(sProcess.options.bidsFolders.Value==3)
            
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
        
        % Store average trial duration - this is needed for model segmentation
        summed_trial_duration = summed_trial_duration + (info_trials(iInput).dataMat.Time(end) - info_trials(iInput).dataMat.Time(1)); % In seconds
        
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
        
        
        %% Gather output filenames
        subject = info_trials(iInput).subject;
        session = info_trials(iInput).session;
        trial   = info_trials(iInput).trial;
        
        for iTrial = 1:length(trial)
        
            if sProcess.options.bidsFolders.Value==1
                % Images
                info_trials(iInput).OutputMriFile{iTrial} = bst_fullfile(['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial{iTrial} '.nii']);
                info_trials(iInput).OutputChannelsFile    = bst_fullfile(['sub-' subject], ['ses-' session], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFile       = bst_fullfile(['sub-' subject], ['ses-' session], 'anat', ['times_' trial{iTrial} '.csv']);

                % Derivatives
                info_trials(iInput).OutputMriFileDerivative{iTrial} = bst_fullfile('derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial{iTrial} '_' annotation '.nii']);
                info_trials(iInput).OutputChannelsFileDerivative    = bst_fullfile('derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFileDerivative       = bst_fullfile('derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['times_' trial{1} '.csv']);

                if iInput==1
                    txt = ['"' trial{iTrial} '"'];
                else
                    txt = [txt ', "' trial{iTrial} '"'];
                end
                
            elseif sProcess.options.bidsFolders.Value==2
                % Images
                subject = [subject session];
                info_trials(iInput).OutputMriFile{iTrial} = bst_fullfile(['sub-' subject], 'anat', ['sub-' subject '_' trial{iTrial} '.nii']);
                info_trials(iInput).OutputChannelsFile    = bst_fullfile(['sub-' subject], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFile       = bst_fullfile(['sub-' subject], 'anat', ['times_' trial{iTrial} '.csv']);

                % Derivatives
                info_trials(iInput).OutputMriFileDerivative{iTrial} = bst_fullfile('derivatives', 'labels', ['sub-' subject], 'anat', ['sub-' subject '_' trial{iTrial} '_' annotation '.nii']);
                info_trials(iInput).OutputChannelsFileDerivative    = bst_fullfile('derivatives', 'labels', ['sub-' subject], 'anat', 'channels.csv');
                info_trials(iInput).OutputTimesFileDerivative       = bst_fullfile('derivatives', 'labels', ['sub-' subject], 'anat', ['times_' trial{1} '.csv']);
                
                if iInput==1
                    txt = ['"' trial{iTrial} '"'];
                else
                    txt = [txt ', "' trial{iTrial} '"'];
                end
                
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
    
    average_trial_duration = summed_trial_duration/length(sInputs);
    
    
    %% Create a figures structure that holds all the window options (FOR PARALLELIZATION PURPOSES - REMOVE)
    
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
            
            % Don't recompute if it already exists (might want to remove that - TODO)
            if ~(exist(bst_fullfile(info_trials(iFile).parentPath, [info_trials(iFile).OutputMriFile{1} '.gz']), 'file') == 2)
                [filenames(iFile), subjects(iFile)] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
            end
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
    
    history_log = gather_trial_history_log({sInputs.FileName});
    
    % Modify config.json file with this dataset's parameters
    if strcmp(sProcess.options.convert.Value, 'conversion')
        modify_config_json(parentPath, modality, annotation, contrast_params_txt, sProcess, {sInputs.FileName}, average_trial_duration, history_log)
    end
    
    
    % === OPEN EXAMPLE IMAGE/DERIVATIVE IN FSLEYES ===
    if sProcess.options.dispExample.Value
        % Check if FSLeyes is installed in the Conda environment
        if ismac
            output=1; % NOT TESTED ON MAC YET -TODO
        elseif isunix
            output = system('fsleyes -h');
            OutputMriFile = bst_fullfile(info_trials(1).parentPath, info_trials(1).OutputMriFile{1});
            OutputMriFileDerivative = bst_fullfile(info_trials(1).parentPath, info_trials(1).OutputMriFileDerivative{1});
            command_to_run = ['fsleyes ' OutputMriFile '.gz -cm render3 ' OutputMriFileDerivative '.gz -cm green --alpha 60 &' ];
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
    
    disp(['BIDS dataset saved in: ' parentPath])
    
end


function [OutputMriFile, subject] = convertTopography2matrix(single_info_trial, sProcess, iFile, figures_struct)

    
    %% ADD A JITTER
    % We want to avoid the model learning the positioning of the event so
    % we crop the time dimension on both sides with a jitter
    
    if strcmp(sProcess.options.convert.Value, 'conversion') && sProcess.options.jitter.Value{1}~=0  % Only during training add a jitter - during (deep learning) segmentation the trial is not truncated
        current_Fs = round(1/diff(single_info_trial.dataMat.Time(1:2)));
        discardElementsBeginning = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);
        discardElementsEnd = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);

        single_info_trial.dataMat.Time = single_info_trial.dataMat.Time(1+discardElementsBeginning:end-discardElementsEnd);
        
        single_info_trial.dataMat.F = single_info_trial.dataMat.F(:,1+discardElementsBeginning:end-discardElementsEnd);
    end
    

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

            % Make a distinction between trials that will be used as baselines
            % (no-annotation - we are just keeping a black NIFTI)
            if ~isempty(iAllSelectedEvents)  % Selected event
                for iSelectedEvent = iAllSelectedEvents
                    annotationValue = 1;
                    isExtended = size(single_info_trial.dataMat.Events(iSelectedEvent).times,1)>1;                                       

                    if isExtended
                        % EXTENDED EVENTS - ANNOTATE BASED ON THEM ONLY
                        for iEvent = 1:size(single_info_trial.dataMat.Events(iSelectedEvent).times,2)
                            iAnnotation_time_edges  = bst_closest(single_info_trial.dataMat.Events(iSelectedEvent).times(:,iEvent)', single_info_trial.dataMat.Time);

                            % If no specific channels are annotated, annotate the entire slice
                            if isempty(single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent})
                                if sProcess.options.gaussian_annot.Value
                                    F_derivative = gaussian_annotation_function(F_derivative, [], iAnnotation_time_edges);
                                else                                    
                                    F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                end
                            else
                                iAnnotation_channels  = find(ismember({single_info_trial.ChannelMat.Channel.Name}, single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                                selectedChannels = iAnnotation_channels;

                                if sProcess.options.gaussian_annot.Value
                                    F_derivative = gaussian_annotation_function(F_derivative, iAnnotation_channels, iAnnotation_time_edges);
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
                            iAnnotation_time_edges  = bst_closest(single_info_trial.dataMat.Events(iSelectedEvent).times(iEvent)+sProcess.options.timewindow_annot.Value{1}, single_info_trial.dataMat.Time);

                            % If no specific channels are annotated, annotate the entire slice
                            if isempty(single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}) || sProcess.options.whole_partial_annotation.Value==1
                                if sProcess.options.gaussian_annot.Value
                                    F_derivative = gaussian_annotation_function(F_derivative, [], iAnnotation_time_edges);
                                else                                    
                                    F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                end
                            else
                                iAnnotation_channels  = find(ismember({single_info_trial.ChannelMat.Channel.Name}, single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                                selectedChannels = iAnnotation_channels;

                                if sProcess.options.gaussian_annot.Value
                                    F_derivative = gaussian_annotation_function(F_derivative, iAnnotation_channels, iAnnotation_time_edges);
                                else                                    
                                    F_derivative(iAnnotation_channels,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                                end
                            end
                        end
                    end
                end
            else
                disp('Baseline trial detected - No annotation on its derivative')
            end

            figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct, modality{iModality});
            [NIFTI_derivative, channels_pixel_coordinates] = channelMatrix2pixelMatrix(F_derivative, single_info_trial.dataMat.Time, single_info_trial.ChannelMat, selectedChannels, iFile, figures_struct, 1, sProcess);
            figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality{iModality});

            if max(max(max(NIFTI_derivative))) ~= 0
                % Set the values to 0 and 1 for the annotations for
                % non-baseline trials (For baseline they are already zero)
                NIFTI_derivative = double(NIFTI_derivative)/max(max(max(double(NIFTI_derivative))));
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
        
            
%         [hFig, iDS, iFig] = view_topography(FileName, Modality, '2DSensorCap');     
        [hFig, iDS, iFig] = view_topography(FileName, Modality, '2DDisc');  
        hFig.Color = [0,0,0]; % This removes nose and ears
%         bst_figures('SetBackgroundColor', hFig, [0 0 0]); % 2DDisc shows a white background - change to black

%         set(hFig, 'Visible', 'off');

        % Remove contour lines
        figure_topo('SetTopoLayoutOptions', 'ContourLines', 0);
        
        AxesHandle = hFig.Children(2);
        pause(.05);

        AxesHandle.PlotBoxAspectRatio = [1,1,1];
        daspect([1 1 1])

%       First make the figure a bit smaller - This is just for looks
        set(hFig, 'Resize', 0);
        set(hFig, 'Position', [hFig.Position(1) hFig.Position(2) 64 64]);
        
        
        % The figure has 2 Children - 1: colorbar, 2: topography
        % Resize topography
        
        
        set(AxesHandle, 'Units', 'pixels', 'Position', [10, 10, 38, 36]);  % Even though I set position [32,32] the getframe returns a 32x32. Leaving it at 34x32 here
%         set(AxesHandle, 'Units', 'pixels', 'Position', [10, 10, 32, 32]);
        
        
        % Find index that just opened figure corresponds to (this is done
        % for enabling parallelization) - COULDNT PARALLELIZE - TODO -
        % MAYBE REMOVE
        all_datafiles = {GlobalData.DataSet.DataFile};
        [temp, index] = ismember(FileName, all_datafiles);
        
        figures_struct(iFile).FigureObject = GlobalData.DataSet(index).Figure;
        figures_struct(iFile).Status       = 'Open';
        figures_struct(iFile).Modality     = Modality;
        

        % Set figure colormap        
        ColormapType = lower(Modality);  % meg, eeg, fnirs
        colormapName = 'gray';
        bst_colormaps('SetColormapName', ColormapType, colormapName);


    elseif strcmp(action, 'close')
        % Close window
%         close(GlobalData.DataSet(iFile).Figure.hFigure)
        close(figures_struct(iFile).FigureObject.hFigure)
        figures_struct(iFile).Status = 'Closed';
    end
end


function [NIFTI, channels_pixel_coordinates] = channelMatrix2pixelMatrix(F, Time, ChannelMat, selectedChannels, iFile, figures_struct, isDerivative, sProcess)

    % GLOBAL MIN_MAX FOR EACH TRIAL
    the_min = min(min(F(selectedChannels,:)));
    the_max = max(max(F(selectedChannels,:)));
    
    % This is altering the EEG 2D display - NOT THE COLORBAR ON THE BOTTOM
%     GlobalData.DataSet(iFile).Figure.Handles.DataMinMax = [the_min, the_max];
    figures_struct(iFile).FigureObject.Handles.DataMinMax = [the_min, the_max];    
    caxis([the_min the_max]);

    % Gets rid of the colorbar object
    delete(figures_struct(iFile).FigureObject.hFigure.Children(1)) 

    img = getframe(figures_struct(iFile).FigureObject.hFigure.Children);
    [height,width,~] = size(img.cdata);
    
    if height~=38 || width~=36
        
        disp(['Forcing resize on the image of trial: ' num2str(iFile) ])
       
        AxesHandle = figures_struct(iFile).FigureObject.hFigure.Children.Children;
        pause(.05);
        set(figures_struct(iFile).FigureObject.hFigure.Children, 'Units', 'pixels', 'Position', [10, 10, 38, 36]);  % Cropping to 32x32 is done later on
       
%         bst_report('Warning', sProcess, sInputs, ['Forced window resizing on trial: ' dataMat.Comment  '.']);

        img.cdata = imresize(img.cdata, [38, 36]);
    end
    img.cdata = img.cdata(5:36,3:34,:); % Crop to 32x32
    
    % For edge effect removal, use this mask
    imageSizeX = width;
    imageSizeY = height;
    [columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
    centerX = 19.5;  % These values are based on the axes dimensions selected on when the figure is resized.
    centerY = 14.5;
    radius = 10.5;
    circlePixels_mask = (rowsInImage - centerX).^2 + (columnsInImage - centerY).^2 <= radius.^2;
    
    
    [height,width,~] = size(img.cdata);

    % Initiate NIFTI matrix
    NIFTI = zeros(height, width, length(Time), 'uint8');
    for iTime = 1:length(Time)
        
        if ~isDerivative
            DataToPlot = F(selectedChannels,iTime);
            iChannel = selectedChannels;
        else
            [tmp,I,J] = intersect(selectedChannels, figures_struct(iFile).FigureObject.SelectedChannels);
%             iChannel = J';
%             set(figures_struct(iFile).FigureObject.hFigure, 'Resize', 1);
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
                DataToPlot_full = full(figures_struct(iFile).FigureObject.Handles.Wmat * DataToPlot);
            % Find first corresponding indices
            else
%                 [tmp,I,J] = intersect(selectedChannels, GlobalData.DataSet(iDS).Figure(iFig).SelectedChannels);
%                 [tmp,I,J] = intersect(selectedChannels, GlobalData.DataSet(iFile).Figure.SelectedChannels);
                [tmp,I,J] = intersect(selectedChannels, figures_struct(iFile).FigureObject.SelectedChannels);
%                 DataToPlot = full(GlobalData.DataSet(iFile).Figure.Handles.Wmat(:,J) * DataToPlot(I));
                DataToPlot_full = full(figures_struct(iFile).FigureObject.Handles.Wmat(:,J) * DataToPlot(I));
            end
            
            % Hack to get the desired value - for some reason it does not
            % work on MEG datasets with Gaussian distribution -It also need to be deactivated when a subselection of channels if defined- TODO
            if isDerivative && length(selectedChannels)>5
                DataToPlot = full(figures_struct(iFile).FigureObject.Handles.Wmat(:,J) * DataToPlot(I));
                DataToPlot_full = ones(size(DataToPlot)) .* F(selectedChannels(1),iTime);
            end
        end         


%         set(GlobalData.DataSet(iFile).Figure.Handles.hSurf, 'FaceVertexCData', DataToPlot, 'EdgeColor', 'none');
        set(figures_struct(iFile).FigureObject.Handles.hSurf, 'FaceVertexCData', DataToPlot_full, 'EdgeColor', 'none');

        % Check exporting image    
        img = getframe(figures_struct(iFile).FigureObject.hFigure.Children);
        [height, width,~] = size(img.cdata);

        if height~=38 || width~=36
           disp(['Forcing resize on the image of trial: ' num2str(iFile) ])
           img.cdata = imresize(img.cdata, [38 36]);
        end
        img.cdata = img.cdata(5:36,3:34,:);
        
        img_gray= rgb2gray(img.cdata);
        
        if isDerivative && all(DataToPlot==0) % This is done since even if all channels are 0, there is still a gray image of the topography displayed to distinguish from the background
            img_gray(img_gray<170)=0;
        elseif isDerivative
            
            if ~sProcess.options.gaussian_annot.Value
                threshold = 150;  % For annotating a single channel assigned it to 133
                img_gray(img_gray<threshold)=0;
                img_gray(img_gray>=threshold)=1;
            end
        end
        
        
        % This gets rids of the pixels that have interpolated values at the
        % edge of the skull.
        remove_edge_effects = 0;
        if remove_edge_effects
            img_gray_new = zeros(size(img_gray));
            img_gray_new(circlePixels_mask) = img_gray(circlePixels_mask);
            img_gray = img_gray_new;
        end
            
        % Assign slice to NIFTI matrix
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

    y_markerLocs = figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1);
    x_markerLocs = figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2);

    
    % HARDCODED
    
%     y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
%     y_in_pixels = 0.95*y_in_pixels;    
    
    y_in_pixels = size(NIFTI,1) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
    y_in_pixels = 2 + 0.97 * y_in_pixels;
    
%     x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
    x_in_pixels = size(NIFTI,2) - size(NIFTI,2) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
    x_in_pixels = 1+ 0.97*x_in_pixels;
%     x_in_pixels =x_in_pixels;

    
    
    
    %% Gather channel coordinates in a struct
    channels_pixel_coordinates = struct;

    for i = 1:length(selectedChannels)
        channels_pixel_coordinates(i).ChannelNames = [ChannelMat.Channel(selectedChannels(i)).Name];
        channels_pixel_coordinates(i).x_coordinates = round(x_in_pixels(i));
        channels_pixel_coordinates(i).y_coordinates = round(y_in_pixels(i));
    end
    
    %%
%     disp(1)
%     h = figure(10);
%     imagesc(squeeze(NIFTI(:,:,75)))
%     colormap('gray')
% 
%     hold on
%     plot(y_in_pixels, x_in_pixels,'*r')
%     hold off

    

    %% Visualize perfect circle from cropping (gets rid of interpolation edge effects)
    
%     figure(1);
%     imagesc(squeeze(NIFTI(:,:,iTime)));colormap gray; title 'Brainstorm topography'
%     hold on
%     plot(y_in_pixels, x_in_pixels,'*r')
%     hold off
%     figure(2);
%     imagesc(circlePixels_mask'); colormap gray; title 'Mask'; colormap gray
%     hold on
%     plot(y_in_pixels, x_in_pixels,'*r')
%     hold off
%     
%     single_slice = squeeze(NIFTI(:,:,iTime))';
%     croppedNIFTI = zeros(size(single_slice));
%     croppedNIFTI(circlePixels_mask) = single_slice(circlePixels_mask);
%     figure(3);
%     imagesc(croppedNIFTI'); colormap gray; title 'Cropped edges'
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


function modify_config_json(parentPath, modality, annotation, contrast_params_txt, sProcess, sInputs_filenames, average_trial_duration, history_log)


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
    config_struct.loader_parameters.path_data = parentPath;
    config_struct.path_output = [parentPath '_output'];
    
    % Change the contrast_params{training_validation, testing}
    contrast_params_cell = cellfun(@(x) strrep(x,'"',''),contrast_params_txt, 'UniformOutput',false);
    config_struct.loader_parameters.contrast_params.training_validation = contrast_params_cell;
    config_struct.loader_parameters.contrast_params.testing = contrast_params_cell;
    
    config_struct.model_name = [modality '_' annotation '_model' ];
    config_struct.loader_parameters.target_suffix = {['_' annotation]};
    
    config_struct.gpu_ids = {0};
    
    % Add the Brainstorm parameters on the json file
    config_struct.brainstorm = struct;
    config_struct.brainstorm.modality = sProcess.options.modality.Comment{sProcess.options.modality.Value};
    config_struct.brainstorm.event_for_ground_truth = sProcess.options.eventname.Value;
    config_struct.brainstorm.annotations_time_window = sProcess.options.timewindow_annot.Value{1};
    config_struct.brainstorm.baselines = logical(sProcess.options.baselines.Value);
    config_struct.brainstorm.annotation = sProcess.options.whole_partial_annotation.Comment{sProcess.options.whole_partial_annotation.Value};
    config_struct.brainstorm.gaussian_annot = logical(sProcess.options.gaussian_annot.Value);
    config_struct.brainstorm.fs = sProcess.options.fs.Value{1};
    config_struct.brainstorm.jitter = sProcess.options.jitter.Value{1};
    config_struct.brainstorm.channel_drop_out = sProcess.options.channelDropOut.Value{1};
    config_struct.brainstorm.bids_folder_creation_mode = sProcess.options.bidsFolders.Comment{sProcess.options.bidsFolders.Value};
    config_struct.brainstorm.average_trial_duration_in_seconds = average_trial_duration;
    
    % Adding an index to help identify the trial
    for i = 1:length(sInputs_filenames)
        sInputs_filenames{i} = [num2str(i) ': ' sInputs_filenames{i}];
    end
    config_struct.brainstorm.sInputs = sInputs_filenames;

    % Add file history
    config_struct.brainstorm.file_history = history_log;

    % Save back to json
    isProspero = 0;  % Prospero is the workstation at McGill. Pretty print fails on it.
                     % Adding pretty print through a python script
    try
        txt = jsonencode(config_struct, 'PrettyPrint', true);
    catch
        txt = jsonencode(config_struct, 'ConvertInfAndNaN', true);
        isProspero = 1;
    end
    
    new_configFile = bst_fullfile(parentPath, 'config_for_training.json');
    fid = fopen(new_configFile, 'w');
    fwrite(fid, txt);
    fclose(fid);    
    
    
    % Pretty print is done on Prospero through a python script.
    % Might change the order of the fields (although I set json.dumps: sort_keys=False)
    if isProspero 
        output = system(['python ' bst_fullfile(bst_get('BrainstormHomeDir'), 'external', 'ivadomed', 'beautify_json.py') ' -f ' new_configFile]);
    end
    
end



function F_derivative = gaussian_annotation_function(F_derivative, iAnnotation_channels, iAnnotation_time_edges)
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
        gaussian_annotation = repmat(single_channel_gaussian_annotation, size(F_derivative,1), 1);
    else
        gaussian_annotation = zeros(size(F_derivative));
        gaussian_annotation_selected_channels = repmat(single_channel_gaussian_annotation, length(iAnnotation_channels), 1);
        
        gaussian_annotation(iAnnotation_channels,:) = gaussian_annotation_selected_channels;
    end
    
    F_derivative = F_derivative + gaussian_annotation;
    F_derivative(F_derivative>1) = 1;

end


function history_log = gather_trial_history_log(trialFileNames)
    history_log = struct;

    for iTrial = 1:length(trialFileNames)
        temp_struct = struct;
        
        sMat = in_bst(trialFileNames{iTrial}, 'History');
        single_file_history = {sMat.History{:,2}; sMat.History{:,3}}';
        single_file_history = cellfun(@cell2mat,num2cell(single_file_history,2),'un',0);
        temp_struct.trialfname = trialFileNames{iTrial};
        temp_struct.history = single_file_history;

        
        history_log(iTrial).trial = temp_struct;
        
    end
    
    disp(1)

end
