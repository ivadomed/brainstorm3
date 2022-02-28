function varargout = process_ivadomed_run( varargin )
% PROCESS_IVADOMED_RUN_IVADOMED: this function calls the ivadomed framework
% with the usage of a config.json file that they define that holds the parameters.
% A flag allows users to bypass certain parameters
% Ivadomed and GPU installations are already assumed.
% https://ivadomed.org/en/latest/index.html

% USAGE:    sProcess = process_run_ivadomed('GetDescription')
%        OutputFiles = process_run_ivadomed('Run', sProcess)
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
    sProcess.Comment     = 'Ivadomed Train/Test Model';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3113;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'import'};
    sProcess.OutputTypes = {'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 0;
    % File selection options
    SelectOptions = {...
        '', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'BIDS dataset input folder...', ...     % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                    % Selection mode: {single,multiple}
        'files', ...                        % Selection mode: {files,dirs,files_and_dirs}
        {{'.json'}, 'Configuration file', 'IVADOMED'}, ... % Available file formats
        []};                               % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    % Option: BIDS folder to train the deep learning model on
    sProcess.options.configFile.Comment = 'Select configuration file:';
    sProcess.options.configFile.Type    = 'filename';
    sProcess.options.configFile.Value   = SelectOptions;
    % File selection options
    SelectOptions = {...
        '', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'BIDS dataset input folder...', ...     % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                    % Selection mode: {single,multiple}
        'dirs', ...                        % Selection mode: {files,dirs,files_and_dirs}
        {{'.folder'}, 'Input BIDS folder', 'IVADOMED'}, ... % Available file formats
        []};                               % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    % Option: BIDS folder to train the deep learning model on
    sProcess.options.BIDSdir.Comment = 'BIDS dataset to train model on:';
    sProcess.options.BIDSdir.Type    = 'filename';
    sProcess.options.BIDSdir.Value   = SelectOptions;
    % File selection options
    SelectOptions = {...
        '', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'Output model folder...', ...     % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                    % Selection mode: {single,multiple}
        'dirs', ...                        % Selection mode: {files,dirs,files_and_dirs}
        {{'.folder'}, 'Ivadomed Output folder', 'IVADOMED'}, ... % Available file formats
        []};                               % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    % Method: Optional Parameters
    sProcess.options.label1.Comment = '<BR><B>Optional Parameters:</B>';
    sProcess.options.label1.Type    = 'label';
    % Option: BIDS folder to train the deep learning model on
    sProcess.options.ivadomedOutputdir.Comment = 'Output Folder for the model:';
    sProcess.options.ivadomedOutputdir.Type    = 'filename';
    sProcess.options.ivadomedOutputdir.Value   = SelectOptions;

    % Method: Command to use
    sProcess.options.label4.Comment = '<BR><B>Command to execute:</B>';
    sProcess.options.label4.Type    = 'label';
    sProcess.options.command.Comment = {'Training', 'Testing', 'Segmentation'};
    sProcess.options.command.Type    = 'radio';
    sProcess.options.command.Value   = 1;
    % Use existing SSPs
    sProcess.options.usessp.Comment = 'Debugging';
    sProcess.options.usessp.Type    = 'checkbox';
    sProcess.options.usessp.Value   = 1;
    % Default selection of components
    sProcess.options.gpu.Comment = 'GPU IDs: ';
    sProcess.options.gpu.Type    = 'value';
    sProcess.options.gpu.Value   = {[0,1,2,3], 'list', 0};
     % Method: Model selection
    sProcess.options.label5.Comment = '<B>Model selection:</B>';
    sProcess.options.label5.Type    = 'label';
    sProcess.options.modelselection.Comment = {'default_model'; 'FiLMedUnet'; 'HeMISUnet'; 'Modified3DUNet'};
    sProcess.options.modelselection.Type    = 'radio';
    sProcess.options.modelselection.Value   = 1;
    % Multichannel
    sProcess.options.multichannel.Comment = 'Multichannel';
    sProcess.options.multichannel.Type    = 'checkbox';
    sProcess.options.multichannel.Value   = 0;
    % Multichannel
    sProcess.options.softgt.Comment = 'Soft groundtruth';
    sProcess.options.softgt.Type    = 'checkbox';
    sProcess.options.softgt.Value   = 0;
    % Method: Average or PCA
    sProcess.options.label6.Comment = '<B>Slice Axis:</B>';
    sProcess.options.label6.Type    = 'label';
    sProcess.options.sliceaxis.Comment = {'Axial'; 'Sagittal'; 'Coronal'};
    sProcess.options.sliceaxis.Type    = 'radio';
    sProcess.options.sliceaxis.Value   = 1;
    % Loss function name
    sProcess.options.loss.Comment = '<B>Loss function:</B>';
    sProcess.options.loss.Type    = 'text';
    sProcess.options.loss.Value   = 'DiceLoss';
    % Uncertainty
    sProcess.options.label7.Comment = '<B>Uncertainty</B>';
    sProcess.options.label7.Type    = 'label';
    sProcess.options.epistemic.Comment = 'Epistemic';
    sProcess.options.epistemic.Type    = 'checkbox';
    sProcess.options.epistemic.Value   = 1;
    sProcess.options.aleatoric.Comment = 'Aleatoric';
    sProcess.options.aleatoric.Type    = 'checkbox';
    sProcess.options.aleatoric.Value   = 0;
    
    % TODO -Trick to fake spikesorter - THIS NEEDS TO BE CHANGED
    sProcess.options.spikesorter.Type   = 'text';
    sProcess.options.spikesorter.Value  = 'ivadomed';
    sProcess.options.spikesorter.Hidden = 1;
    % Options: Options
    sProcess.options.edit.Comment = {'panel_spikesorting_options', '<U><B>Config file</B></U>: '};
    sProcess.options.edit.Type    = 'editpref';
    sProcess.options.edit.Value   = [];
    
    
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess)
    Comment = 'Ivadomed Train/Test Model';
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
    
    
    %% Do some checks on the parameters
    
    if strcmp(sProcess.options.convert.Value, 'conversion')
    
        % In case the selected event is single event, or the eventname isempty,
        % make sure the time-window has values
        inputs_to_remove = false(length(sInputs),1);

        for iInput = 1:length(sInputs)
            dataMat = in_bst(sInputs(iInput).FileName, 'Events');
            events = dataMat.Events;

            Time = in_bst(sInputs(iInput).FileName, 'Time');

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
                    if sProcess.options.timewindow.Value{1}(1)<Time(1) || sProcess.options.timewindow.Value{1}(2)>Time(end)
                        inputs_to_remove(iInput) = true;
                        bst_report('Warning', sProcess, sInputs, ['The time window selected for annotation is not within the Time segment of trial: ' dataMat.Comment ' . Ignoring this trial']);                    
                    end

                end
            end
        end

        % The following works on Matlab R2021a - I think older versions
        % need another command to squeeze the empty structure entries - TODO
        sInputs(inputs_to_remove) = [];
    end
    
    %% Gather F  and filenames for all files here - This shouldn't create a memory issue
    % Reason why all of them are imported here is that in_bst doesn't like
    % to be parallelized (as far as I checked), so can't call it within 
    % convertTopography2matrix
    
    empty_IVADOMED_folder = true;
    
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       protocol.Comment);
                   
    % Remove previous entries
    if empty_IVADOMED_folder
        if exist(parentPath, 'dir') 
            rmdir(parentPath, 's')
        end
        if exist([parentPath '-meta'], 'dir') 
            rmdir([parentPath '-meta'], 's')
        end
%         if exist([parentPath '-segmentation'], 'dir') 
%             rmdir([parentPath '-segmentation'], 's')
%         end
%         if exist([parentPath '-segmentation-meta'], 'dir') 
%             rmdir([parentPath '-segmentation-meta'], 's')
%         end
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
    
%     txt  = [];
    
    for iInput = 1:length(sInputs)
        info_trials(iInput).FileName = sInputs(iInput).FileName;
        info_trials(iInput).subject = lower(str_remove_spec_chars(sInputs(iInput).SubjectName));
        info_trials(iInput).session = lower(str_remove_spec_chars(sInputs(iInput).Condition));
        
        % The trial # needs special attention
        splitComment = split(sInputs(iInput).Comment,{'(#',')'});
        comment = lower(str_remove_spec_chars(splitComment{1}));
        iEpoch = str2double(splitComment{2});
        
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
        info_trials(iInput).trial = [comment iEpoch];  % TODO - THE UNIQUE LIST OF TRIAL LABELS NEEDS TO BE ADDED ON THE IVADOMED CONFIG.JSON FILE TO BE INCLUDED IN TRAINING/VALIDATION
      
        % Load data structure
        info_trials(iInput).dataMat = in_bst(sInputs(iInput).FileName);
        
        
        % And resample if needed
        current_Fs = round(1/diff(info_trials(iInput).dataMat.Time(1:2)));
        if ~isnan(wanted_Fs) && current_Fs~=wanted_Fs
            %[x, time_out] = process_resample('Compute', x, time_in, NewRate)
            [info_trials(iInput).dataMat.F, info_trials(iInput).dataMat.Time] = process_resample('Compute', info_trials(iInput).dataMat.F, info_trials(iInput).dataMat.Time, wanted_Fs);
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
        
        
        % Get output filename
        subject = info_trials(iInput).subject;
        session = info_trials(iInput).session;
        trial   = info_trials(iInput).trial;
        
        if sProcess.options.bidsFolders.Value==1
            % Images
            info_trials(iInput).OutputMriFile      = bst_fullfile(parentPath, ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial '.nii']);
            info_trials(iInput).OutputChannelsFile = bst_fullfile(channels_times_path, ['sub-' subject], ['ses-' session], 'anat', 'channels.csv');
            info_trials(iInput).OutputTimesFile    = bst_fullfile(channels_times_path, ['sub-' subject], ['ses-' session], 'anat', ['times_' trial '.csv']);
            
            % Derivatives
            info_trials(iInput).OutputMriFileDerivative      = bst_fullfile(parentPath, 'derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial '_' annotation '.nii']);
            info_trials(iInput).OutputChannelsFileDerivative = bst_fullfile(channels_times_path, 'derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', 'channels.csv');
            info_trials(iInput).OutputTimesFileDerivative    = bst_fullfile(channels_times_path, 'derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['times_' trial '.csv']);
            
        elseif sProcess.options.bidsFolders.Value==2
            % Images
            subject = [subject session];
            info_trials(iInput).OutputMriFile      = bst_fullfile(parentPath, ['sub-' subject], 'anat', ['sub-' subject '_' trial '.nii']);
            info_trials(iInput).OutputChannelsFile = bst_fullfile(channels_times_path, ['sub-' subject], 'anat', 'channels.csv');
            info_trials(iInput).OutputTimesFile    = bst_fullfile(channels_times_path, ['sub-' subject], 'anat', ['times_' trial '.csv']);
            
            % Derivatives
            info_trials(iInput).OutputMriFileDerivative      = bst_fullfile(parentPath, 'derivatives', 'labels', ['sub-' subject], 'anat', ['sub-' subject '_' trial '_' annotation '.nii']);
            info_trials(iInput).OutputChannelsFileDerivative = bst_fullfile(channels_times_path, 'derivatives', 'labels', ['sub-' subject], 'anat', 'channels.csv');
            info_trials(iInput).OutputTimesFileDerivative    = bst_fullfile(channels_times_path, 'derivatives', 'labels', ['sub-' subject], 'anat', ['times_' trial '.csv']);
        elseif sProcess.options.bidsFolders.Value==3
            % Images
            subject = [subject num2str(ii)];
            info_trials(iInput).subject            = subject;
            info_trials(iInput).OutputMriFile      = bst_fullfile(parentPath, ['sub-' subject], 'anat', ['sub-' subject '_' trial '.nii']);
            info_trials(iInput).OutputChannelsFile = bst_fullfile(channels_times_path, ['sub-' subject], 'anat', 'channels.csv');
            info_trials(iInput).OutputTimesFile    = bst_fullfile(channels_times_path, ['sub-' subject], 'anat', ['times_' trial '.csv']);

            % Derivatives
            info_trials(iInput).OutputMriFileDerivative      = bst_fullfile(parentPath, 'derivatives', 'labels', ['sub-' subject], 'anat', ['sub-' subject '_' trial '_' annotation '.nii']);
            info_trials(iInput).OutputChannelsFileDerivative = bst_fullfile(channels_times_path, 'derivatives', 'labels', ['sub-' subject], 'anat', 'channels.csv');
            info_trials(iInput).OutputTimesFileDerivative    = bst_fullfile(channels_times_path, 'derivatives', 'labels', ['sub-' subject], 'anat', ['times_' trial '.csv']);
            ii = ii + 1;
            
%             txt = [txt ', "' trial '"'];
        end
    end
    
    
    %% Prepare parallel pool, if requested
    if sProcess.options.paral.Value
        try
            poolobj = gcp('nocreate');
            if isempty(poolobj)
                parpool;
            end
        catch
            sProcess.options.paral.Value = 0;
            poolobj = [];
        end
    else
        poolobj = [];
    end
    
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
    filenames = cell(length(info_trials),1);
    subjects = cell(length(info_trials),1);
    
    
    start_time = tic;
    if isempty(poolobj)
        bst_progress('start', 'Ivadomed', 'Converting trials to NIFTI files...', 0, length(sInputs));
        for iFile = 1:length(info_trials)
            disp(num2str(iFile))
            [filenames{iFile}, subjects{iFile}] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
            bst_progress('inc', 1);
        end
    else
        bst_progress('start', 'Ivadomed', 'Converting trials to NIFTI files...', 0, 0);
        parfor iFile = 1:length(info_trials)
            [filenames{iFile}, subjects{iFile}] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
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
    
    
    if sProcess.options.dispExample.Value
        % === OPEN EXAMPLE IMAGE/DERIVATIVE IN FSLEYES ===
        % Check if FSLeyes is installed in the Conda environment
        output = system('fsleyes -h');
        command_to_run = ['fsleyes ' info_trials(1).OutputMriFile '.gz -cm render3 ' info_trials(1).OutputMriFileDerivative '.gz -cm green --alpha 60' ];
        if output~=0
            bst_report('Warning', sProcess, sInputs(1), 'Fsleyes package is not accessible. Are you running Matlab through an anaconda environment that has Fsleyes installed?');
            disp(command_to_run)
            return
        else
            system(['fsleyes ' info_trials(1).OutputMriFile '.gz -cm render3 ' info_trials(1).OutputMriFileDerivative '.gz -cm green --alpha 50' ]);
        end
    end
    
    
end


function [OutputMriFile, subject] = convertTopography2matrix(single_info_trial, sProcess, iFile, figures_struct)
%   % Ignoring the bad sensors in the interpolation, so some values will be interpolated from the good sensors
%   WExtrap = GetInterpolation(iDS, iFig, TopoInfo, Vertices, Faces, bfs_center, bfs_radius, chan_loc(selChan,:));
% 
%         
%   [DataToPlot, Time, selChan, overlayLabels, dispNames, StatThreshUnder, StatThreshOver] = GetFigureData(iDS, iFig, 0);

    % TODO
    disp('DOES The colormap need to be GRAY???') % ANSWER: YES
        
    subject = single_info_trial.subject;
    session = single_info_trial.session;
    trial   = single_info_trial.trial;
    
    
    %modality = sProcess.options.modality.Comment{sProcess.options.modality.Value};
    
    
    modality = 'MEG'
    
    
    
    
    
    
    
    
        figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct, modality);

    
    
    
    
    
    
    
    
    
    
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
        
        
        
        
        
        
        
        
        
        %%
    
    
    
    
    
    
    %% ADD A JITTER
    % We want to avoid the model learning the positioning of the event so
    % we crop the time dimension on both sides with a jitter
    
    if strcmp(sProcess.options.convert.Value, 'conversion')  % Only during training add a jitter - during (deep learning) segmentation the trial is not truncated
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
    
%     %% CHANGE THE COLORMAP DISPLAYED ON THE BOTTOM LEFT
%     % Get colormap type
%     ColormapInfo = getappdata(GlobalData.DataSet.Figure.hFigure, 'Colormap');
%     
%     % ==== Update colormap ====
%     % Get colormap to use
%     sColormap = bst_colormaps('GetColormap', ColormapInfo.Type);
%     % Set figure colormap (for display of the colorbar only)
%     set(GlobalData.DataSet.Figure.hFigure, 'Colormap', sColormap.CMap);
%     set(GlobalData.DataSet.Figure.hFigure, 'Colormap', sColormap.CMap);
    

    % Select the appropriate sensors
    nElectrodes = 0;
    selectedChannels = [];
    for iChannel = 1:length(single_info_trial.ChannelMat.Channel)
       if (strcmp(single_info_trial.ChannelMat.Channel(iChannel).Type, 'EEG')  || ...
           strcmp(single_info_trial.ChannelMat.Channel(iChannel).Type, 'MEG')) && ...
           single_info_trial.dataMat.ChannelFlag(iChannel)==1
          
               %% TODO - ACCOMMODATE MORE HERE - fNIRS?
          nElectrodes = nElectrodes + 1;
          selectedChannels(end + 1) = iChannel;
       end
    end
    
    %% Gather the topography slices to a single 3d matrix
    % Here the time dimension is the 3rd dimension
    [NIFTI, channels_pixel_coordinates] = channelMatrix2pixelMatrix(single_info_trial.dataMat.F, single_info_trial.dataMat.Time, single_info_trial.ChannelMat, selectedChannels, iFile, figures_struct, 0);
%     figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality);

    %% Get the output filename
    
    % Substitute the voxels with the 2D slices created from the 2dlayout
    % topography
    single_info_trial.sMri.Cube = NIFTI;
    
    %% Export the created cube to NIFTI
    OutputMriFile = export2NIFTI(single_info_trial.sMri, single_info_trial.OutputMriFile);
    
    %% Export times if doing training, and channels' coordinates if doing segmentation
    export_Channels_Times(single_info_trial, channels_pixel_coordinates, single_info_trial.dataMat.Time', sProcess.options.convert.Value);
      
    
    %% Create derivative

    % The derivative will be based on a period of time that is annotated to
    % be the Ground truth.
    % In the case of extended event, only that period of time will annotated
    % In the case of simple event, the selection 
    
    % First check that an event label was selected by the user for
    % annotation
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
                            F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                        else
                            iAnnotation_channels  = find(ismember({single_info_trial.ChannelMat.Channel.Name}, single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                            F_derivative(iAnnotation_channels,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                            selectedChannels = iAnnotation_channels;
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
                            F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                        else
                            iAnnotation_channels  = find(ismember({single_info_trial.ChannelMat.Channel.Name}, single_info_trial.dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                            F_derivative(iAnnotation_channels,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
                            selectedChannels = iAnnotation_channels;
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

        figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct, modality);
        [NIFTI_derivative, channels_pixel_coordinates] = channelMatrix2pixelMatrix(F_derivative, single_info_trial.dataMat.Time, single_info_trial.ChannelMat, selectedChannels, iFile, figures_struct, 1);
        figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality);


        % Set the values to 0 and 1 for the annotations
        % Hard threshold
        NIFTI_derivative = NIFTI_derivative/max(max(max(NIFTI_derivative)));

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

        % Annotate derivative
        single_info_trial.sMri.Cube = NIFTI_derivative;

        %% Export the created cube to NIFTI
        export2NIFTI(single_info_trial.sMri, single_info_trial.OutputMriFileDerivative);

%         %% Export the channel coordinates to a .csv file
%         writetable(struct2table(channels_pixel_coordinates), single_info_trial.OutputChannelsFileDerivative, 'Delimiter', '\t')

%         %% Export times to a .csv (This is needed since the trials have been truncated from the JITTER parameter)
%         writematrix(single_info_trial.dataMat.Time', single_info_trial.OutputTimesFileDerivative)
    else
    	figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct, modality);
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
        figures_struct(iFile).Status = 'Open';
        

        %TODO - GET COLORMAP TYPE AUTOMATICALLY (meg - eeg - fnirs)
        % Get figure colormap
        
        ColormapType = lower(Modality);  % meg, eeg, fnirs
        colormapName = 'gray';
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


function [NIFTI, channels_pixel_coordinates] = channelMatrix2pixelMatrix(F, Time, ChannelMat, selectedChannels, iFile, figures_struct, isDerivative)
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
            
            threshold = 145;  % For annotating a single channel assigned it to 133
            img_gray(img_gray<threshold)=0;
            img_gray(img_gray>=threshold)=1;
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
        % MONITOR
        % HARDCODED CHANNEL POSITION CORRECTION - TODO
        y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
        y_in_pixels = y_in_pixels/0.84;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
        x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
        x_in_pixels = 2 + x_in_pixels/1.14;
    elseif all(Pix_SS == [1 1 1920 1080])
        % LAPTOP
        % HARDCODED CHANNEL POSITION CORRECTION - TODO
        y_in_pixels = pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
        y_in_pixels = 0 + y_in_pixels/0.836;  % The axis ratio needs to be [1,1,1] TODO - to remove hardcoded entry     
        x_in_pixels = pos(4) - pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));  % Y axis is reversed, so I subtract from pos(4)
        x_in_pixels = 1 + x_in_pixels/1.23;
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
%     imagesc(squeeze(NIFTI(:,:,1)))
%     colormap('gray')
% 
%     hold on
%     plot(y_in_pixels, x_in_pixels,'*r')
%     plot(y_in_pixels(iChannel), x_in_pixels(iChannel),'*g')
%     hold off

    
end


function OutputMriFile = export2NIFTI(sMri, OutputMriFile)
    %% Export to NIFTI

    % Create the output folder first
    [filepath,name,ext] = fileparts(OutputMriFile);
    if ~exist(filepath, 'dir')  % This avoids a warning if the folder already exists
        mkdir(bst_fileparts(OutputMriFile))
    end    

    % Export (.nii)
    out_mri_nii(sMri, OutputMriFile);
    
    % Zip nifti (create .nii.gz)
    gzip(OutputMriFile)
    % Delete nifti (.nii)
    delete(OutputMriFile)
    
    OutputMriFile = [OutputMriFile '.gz'];
    
end


function export_Channels_Times(single_info_trial, channels_pixel_coordinates, Time, method)

    switch method
        case 'conversion' 
            % For segmentation we dont really need this - we have the info from the
            % trials themselved - they are not truncated as is the case with the
            % training part
            % Create the output folder first
            [filepath,name,ext] = fileparts(single_info_trial.OutputTimesFile);
            if ~exist(filepath, 'dir')  % This avoids a warning if the folder already exists
                mkdir(bst_fileparts(single_info_trial.OutputChannelsFile))
            end
            
            % Export times to a .csv (This is needed since the trials have been truncated from the JITTER parameter)
            writematrix(Time, single_info_trial.OutputTimesFile)
            
        case 'segmentation'
            % Create the output folder first
            [filepath,name,ext] = fileparts(single_info_trial.OutputChannelsFile);
            if ~exist(filepath, 'dir')  % This avoids a warning if the folder already exists
                mkdir(bst_fileparts(single_info_trial.OutputChannelsFile))
            end    

            % Export the channel coordinates to a .csv file
            writetable(struct2table(channels_pixel_coordinates), single_info_trial.OutputChannelsFile, 'Delimiter', '\t')
        
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

