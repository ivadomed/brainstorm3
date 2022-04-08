function varargout = process_ivadomed_segmentation( varargin )
% PROCESS_IVADOMED_SEGMENTATION: Use pretrained model for segmentation on
% the selected raw signals or trials
%
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
% Authors: Konstantinos Nasiotis 2021-2022

eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    sProcess.Comment     = 'Ivadomed Segmentation';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3114;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'raw', 'data'};
    sProcess.OutputTypes = {'raw', 'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    
    
    sProcess.options.label1.Comment = '<B>Link to raw files options<I><FONT color="#777777">  (Ignore for trial inputs)</FONT></I></B>';
    sProcess.options.label1.Type    = 'label';
    % Recordings time window
    sProcess.options.timewindow.Comment = 'Recordings Time window:';
    sProcess.options.timewindow.Type    = 'timewindow';
    sProcess.options.timewindow.Value   = [];
    % Sliding window 
    sProcess.options.slidingWindowOverlap.Comment = 'Sliding windows overlap';
    sProcess.options.slidingWindowOverlap.Type    = 'value';
    sProcess.options.slidingWindowOverlap.Value   = {10, '%', 0};
    % FILTERING OPTIONS
    % === Low bound
    sProcess.options.highpass.Comment = 'Lower cutoff frequency (0=disable):';
    sProcess.options.highpass.Type    = 'value';
    sProcess.options.highpass.Value   = {0.5,'Hz ',3};
    % === High bound
    sProcess.options.lowpass.Comment = 'Upper cutoff frequency (0=disable):';
    sProcess.options.lowpass.Type    = 'value';
    sProcess.options.lowpass.Value   = {70,'Hz ',3};
    
    SelectOptions = {...
        '', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'Output model folder...', ...     % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                    % Selection mode: {single,multiple}
        'dirs', ...                        % Selection mode: {files,dirs,files_and_dirs}
        {{'.folder'}, 'Model output folder', 'IVADOMED'}, ... % Available file formats
        []};                               % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    
%     % Model selection options
%     SelectOptions = {...
%         '', ...                               % Filename
%         '', ...                               % FileFormat
%         'open', ...                           % Dialog type: {open,save}
%         'Import model...', ...               % Window title
%         'ImportData', ...                     % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
%         'single', ...                         % Selection mode: {single,multiple}
%         'files', ...                          % Selection mode: {files,dirs,files_and_dirs}
%         bst_get('FileFilters', 'model', 'IVADOMED'), ... % Get all the available file formats
%         'EventsIn'};                          % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn
    
    
    % Deep learning model
    sProcess.options.label2.Comment = '<B>Ivadomed trained model</B>';
    sProcess.options.label2.Type    = 'label';
    % Option: Model folder
    sProcess.options.modelfolder.Comment = 'Deep learning model output folder:';
    sProcess.options.modelfolder.Type    = 'filename';
    sProcess.options.modelfolder.Value   = SelectOptions;
%     % Option: Model file
%     sProcess.options.modelfile.Comment = 'Deep learning model file:';
%     sProcess.options.modelfile.Type    = 'filename';
%     sProcess.options.modelfile.Value   = SelectOptions;
    % Newly created Event name
    sProcess.options.eventname.Comment = 'Label for annotated event: ';
    sProcess.options.eventname.Type    = 'text';
    sProcess.options.eventname.Value   = 'segmented';
    % GPU ID to run inference on
    sProcess.options.gpu.Comment = 'GPU ID to run inference on:';
    sProcess.options.gpu.Type    = 'value';
    sProcess.options.gpu.Value   = {0, [], 0};
    % Conversion of both trials and their derivatives
    sProcess.options.convert.Type   = 'text';
    sProcess.options.convert.Value  = 'segmentation';  % Other option: 'conversion'
    sProcess.options.convert.Hidden = 1;
    % BIDS subject selection
    % Method: BIDS subject selection
    sProcess.options.annotationLabel.Comment = '<I><FONT color="#FF0000">Whole Head annotation (all channels) or partial (annotate specific channels)</FONT></I>';
    sProcess.options.annotationLabel.Type    = 'label';
    sProcess.options.annotation.Comment = {'Whole', 'Partial'};
    sProcess.options.annotation.Type    = 'radio';
    sProcess.options.annotation.Value   = 1;
    % GPU ID to run inference on - % this allows to allocate the percentage
    % of channels that need to have the annotation in order to keep the
    % annotation
    sProcess.options.majorityVote.Comment = 'Majority Vote Percentage of MEG/EEG channels [0,100]</FONT></I>';
    sProcess.options.majorityVote.Type    = 'value';
    sProcess.options.majorityVote.Value   = {50, [], []};
    % Parallel processing
    sProcess.options.paral.Comment = 'Parallel processing';
    sProcess.options.paral.Type    = 'checkbox';
    sProcess.options.paral.Value   = 0;
    sProcess.options.paral.Hidden = 1;
    % Display example image in FSLeyes
    sProcess.options.dispExample.Comment = 'Open an example image/derivative on FSLeyes';
    sProcess.options.dispExample.Type    = 'checkbox';
    sProcess.options.dispExample.Value   = 0;
    sProcess.options.dispExample.Hidden  = 1;
    
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = ['Detect: ', sProcess.options.eventname.Value];
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>   
    
    %% Check if IVADOMED is installed/accessible
    % We'll check for two options:
    % 1. Ivadomed is installed through the plugin manager
    % 2. Ivadomed is accessible as a system call but not as a plugin
    
    PlugDesc = bst_plugin('GetInstalled','ivadomed');
    
    if isempty(PlugDesc)
        isLoaded = 0;
    else
        isLoaded = PlugDesc.isLoaded;
    end

    fname = bst_fullfile(PlugDesc.Path, 'ivadomed-master', 'ivadomed', 'main.py');
    
    if ~isLoaded
        
        % Check if ivadomed can be accessed from a system call
        % in case the user installed it outside of Brainstorm
        output = system('ivadomed -h');
        if output~=0
            bst_report('Error', sProcess, sInputs, 'Ivadomed package is not accessible. Check if it is installed as a plugin or if you are running Matlab through an anaconda environment that has Ivadomed installed');
            OutputFiles = {};
            return
        else
            ivadomed_call = 'ivadomed';
        end
    else
        % Call to be used as a system command when calling ivadomed
        ivadomed_call = ['python3 ' fname];
    end
    

    %% ===== GET OPTIONS =====
    % Event name
    evtName = strtrim(sProcess.options.eventname.Value);
    modelFolder = sProcess.options.modelfolder.Value{1};
    
    if isempty(evtName) || isempty(modelFolder)
        bst_report('Error', sProcess, [], 'Event name and trained model must be specified.');
        OutputFiles = {};
        return;
    end
    
    %% Important files/folders
    ivadomedOutputFolder = modelFolder;  % Output of the trained model
    
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '-segmentation']);
    channelsparentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '-meta-segmentation']);
    
    %% CHANGE THE CONFIG FILE TO RUN LOCALLY
    
    % Grab the config.json file that was used and assign the gpu that the
    % user selected
    configFile = bst_fullfile(modelFolder, 'config_file.json');
    
    fid = fopen(configFile);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);          
    
    %Substitute null values with nan - This is needed for cause jsonencode
    %changes null values to [] and ultimately ivadomed throws errors
    str = strrep(str, 'null', 'NaN');
        
    config_struct = jsondecode(str);
    
    %config_struct.command = 'segment'; % Not needed - use CLI call
    config_struct.gpu_ids = {sProcess.options.gpu.Value{1}};
    
    % Save back to json
%     txt = jsonencode(config_struct, 'PrettyPrint', true);
    txt = jsonencode(config_struct, 'ConvertInfAndNaN', true);
    
    fid = fopen(configFile, 'w');
    fwrite(fid, txt);
    fclose(fid);

    
    %% Assign some values from the config file
    sProcess.options.fs.Value               = {config_struct.brainstorm.fs, 'Hz', 0};
    sProcess.options.timewindow_annot.Value = {config_struct.brainstorm.annotations_time_window, 'ms', []};
    sProcess.options.modality.Comment       = {'MEG', 'EEG', 'MEG+EEG', 'fNIRS'}; % Make sure that the order of the modalities doesn't change between this function and process_ivadomed_create_dataset.m
    sProcess.options.modality.Value         = find(ismember(sProcess.options.modality.Comment, config_struct.brainstorm.modality));  
    sProcess.options.bidsFolders.Comment    = {'Normal', 'Separate runs/sessions as different subjects', 'Separate each trial as different subjects'}; % Same
    sProcess.options.bidsFolders.Value      = find(ismember(sProcess.options.bidsFolders.Comment, config_struct.brainstorm.bids_folder_creation_mode));  
    sProcess.options.channelDropOut.Value   = {config_struct.brainstorm.channel_drop_out, 'channels', 0};
    
    %% If the input is a raw file, segment it into "trials" with a sliding window
    % The size of the window needs to be the same as the trials used to
    % train the model - TODO
    
    
    % If the first input is raw, all of them are. Brainstorm doesnt allow 
    % inputs of both raw and trials
    
    [sStudy, iStudy, iData] = bst_get('DataFile', sInputs(1).FileName);
    isRaw = strcmpi(sStudy.Data(iData).DataType, 'raw');
    
    
    if isRaw % TODO - RIGHT NOW IT IS DONE ONLY FOR A SINGLE INPUT LINK TO RAW FILE - CHANGE TO MULITPLE
        ImportOptions.ImportMode = 'Event';
        ImportOptions.UseEvents = 1;
        ImportOptions.EventsTimeRange = [0 config_struct.brainstorm.average_trial_duration_in_seconds];
        ImportOptions.GetAllEpochs = 0;
        ImportOptions.iEpochs = 1;
        ImportOptions.SplitRaw = 0;
        ImportOptions.SplitLength = [];
        ImportOptions.UseCtfComp = 1; % PROBABLY THIS IS ONLY FOR CTF - TODO
        ImportOptions.UseSsp = 1;
        ImportOptions.RemoveBaseline = 'all';
        ImportOptions.BaselineRange = [];
        ImportOptions.CreateConditions = 1;
        ImportOptions.ChannelReplace = 1;
        ImportOptions.ChannelAlign = 1;
        ImportOptions.IgnoreShortEpochs = 1;
        ImportOptions.EventsMode = 'ask';
        ImportOptions.EventsTrackMode = 'ask';
        ImportOptions.EventsTypes = '';
        ImportOptions.DisplayMessages = 0;  % This prevents the importing GUI to pop up
        ImportOptions.Precision = [];

        % TODO
        if length(sInputs)>1
            bst_error('Multiple link to raw file inputs not supported yet. Drop a single link to raw file to the processing box')
        end
        
        for iInput = 1:length(sInputs) % Link to raw files inputs

    %         [DataFile_path, DataFile_base] = bst_fileparts(DataFile);
            [DataFile_path, DataFile_base] = bst_fileparts(sInputs(iInput).FileName);
            [sStudy, iStudy, iData] = bst_get('DataFile', sInputs(iInput).FileName);
            [sSubject, iSubject] = bst_get('Subject', sStudy.BrainStormSubject);

            [sStudy, iStudy] = bst_get('StudyWithCondition', DataFile_path);
    %         [sStudy, iStudy] = bst_get('StudyWithCondition', bst_fullfile(sSubject.Name, Condition));


            % Read file descriptor
            DataMat = in_bst_data(sInputs(iInput).FileName);
            % Read channel file
            ChannelFile = bst_get('ChannelFileForStudy', sInputs(iInput).FileName);
            ChannelMat = in_bst_channel(ChannelFile);
            % Get sFile structure
            if isRaw
                sFile = DataMat.F;
            else
                sFile = in_fopen(sInputs(iInput).FileName, 'BST-DATA');
            end


            % Create a fake event that corresponds to the sliding windows and their
            % selected overlap

            if isempty(sProcess.options.timewindow.Value{1})
                timeRange = sFile.prop.times;
            else
                timeRange = sProcess.options.timewindow.Value{1};
                % In case the selection is outside of the time vector, reassign
                if timeRange(1)<sFile.prop.times(1)
                    timeRange(1)=sFile.prop.times(1);
                end
                if timeRange(2)>sFile.prop.times(2)
                    timeRange(2)=sFile.prop.times(2);
                end
            end

            ImportOptions.TimeRange = timeRange;

            % Check if it is needed to resample
            if config_struct.brainstorm.fs~=sFile.prop.sfreq
                ImportOptions.Resample = 1;
                ImportOptions.ResampleFreq = config_struct.brainstorm.fs;
            else
                ImportOptions.Resample = 0;
                ImportOptions.ResampleFreq = [];
            end

            % Get the starting timepoints of each window
            slidingWindowTimes = timeRange(1):config_struct.brainstorm.average_trial_duration_in_seconds*(100-sProcess.options.slidingWindowOverlap.Value{1})/100:timeRange(2);

            newEvents.label      = 'slidingWindow';
            newEvents.color      = [rand(1,1), rand(1,1), rand(1,1)];
            newEvents.times      = slidingWindowTimes;
            newEvents.reactTimes = [];
            newEvents.select     = 1;
            newEvents.epochs     = ones(1, size(newEvents(1).times, 2));
            newEvents.channels   = cell(1, size(newEvents(1).times, 2));
            newEvents.notes      = cell(1, size(newEvents(1).times, 2));

            ImportOptions.events = newEvents;

            % Import the trials in the database
            % NOTE: The output does not include the BAD trials
    %         NewFiles = import_data(DataFiles, ChannelMat, FileFormat, iStudyInit, iSubjectInit, ImportOptions, DateOfStudy)
    %         NewFiles = import_data(sFile,     ChannelMat, FileFormat, iStudyInit, iSubjectInit, ImportOptions, DateOfStudy=[])
            
            NewFiles = import_data(sFile, ChannelMat, sFile.format, iStudy, iSubject, ImportOptions, []);  % This is actually creating new database entries. Consider saving in the tmp folder - TODO
        end
        
        % Get the "sInputs structure" of the trials so it can be used as an
        % input to process_ivadomed_create_dataset 
        sInputs_trials = bst_process('GetInputStruct', NewFiles); % should be sInputs_trials{iInput} - TODO
        
        % Find which point in time of the recording is sliding window
        % starts at. I could only find this info being stored at the file's
        % history. Might need to revisit this - TODO
        slidingWindowTimes = zeros(length(NewFiles),1);
        
        for iTrial = 1:length(NewFiles)
            trial_info = in_bst(NewFiles{iTrial}, 'History');
            iRawHistory = find(ismember({trial_info.History{:,2}}, 'import_time'));
            temp_segment_boundaries = strsplit(trial_info.History{iRawHistory,3},{'[',',',']'});
            slidingWindowTimes(iTrial) = str2double(temp_segment_boundaries{2});
        end
        
        
        
    else
        sInputs_trials = sInputs;
        
        % Add a warning if the trials are not the size of the trials that
        % were used for training the selected model
        
        iPotentially_problematic_trials = [];
        
        for iInput = 1:length(sInputs)
            time = in_bst(sInputs_trials(iInput).FileName, 'Time');
            
            time_length = time(end) - time(1);
            if time_length > config_struct.brainstorm.average_trial_duration_in_seconds
                iPotentially_problematic_trials = [iPotentially_problematic_trials, iInput];
            end
        end

        bst_report('Warning', sProcess, sInputs(iPotentially_problematic_trials), ['These trials are longer than ' num2str(config_struct.brainstorm.average_trial_duration_in_seconds) ...
                            ' seconds,  which is the size of the trials that the model was trained on, and the deep learning segmentation might be inaccurate. Consider creating smaller trials.']);

        
        
    end
    
    
    %% Create a BIDS dataset with the trials to be segmented
    get_filenames = 1;
    OutputFiles = process_ivadomed_create_dataset('Run', sProcess, sInputs_trials, get_filenames); % TODO - RIGHT NOW IT IS DONE ONLY FOR A SINGLE INPUT LINK TO RAW FILE - CHANGE TO MULITPLE
    
    disp('% TODO - RIGHT NOW IT IS DONE ONLY FOR A SINGLE INPUT LINK TO RAW FILE - CHANGE TO MULITPLE')
    %% Call ivadomed with "segment" method
    output = system([ivadomed_call ' --segment -c ' configFile ' -pd ' parentPath ' -po ' ivadomedOutputFolder]);
    if output~=0
        bst_report('Error', sProcess, sInputs_trials, 'Something went wrong during segmentation');
        return
    end
    
    %% Get ivadomed output segmentation files
    segmentationMasks = cell(length(sInputs_trials),1);
    for iInput = 1:length(sInputs_trials)
        [a,b,c] = bst_fileparts(OutputFiles{iInput});
        [a,file_basename,c] = bst_fileparts(b);

        % Grab event annotation suffix (e.g. "centered") from json file
%         segmentationMasks{iInput} = bst_fullfile(ivadomedOutputFolder, 'pred_masks', [file_basename, config_struct.loader_parameters.target_suffix{1}, '_pred.nii.gz']);
        segmentationMasks{iInput} = bst_fullfile(ivadomedOutputFolder, 'pred_masks', [file_basename, '_class-0', '_pred.nii.gz']);
    end
    
    %% Converter from masks to Brainstorm events
    nEvents = 0;
    nTotalOcc = 0;
    
    link_to_raw_file_events_times = []; % This is used only if the link to raw files are used
    
    for iInput = 1:length(sInputs_trials)
        MriFile = segmentationMasks{iInput};
        unzippedMask = gunzip(MriFile);

        [MRI, vox2ras] = in_mri_nii(unzippedMask{1}, 1, 1, 0);

        % Delete nifti (.nii) - keep the unzipped for debugging - TODO -
        % consider removing the .nii.gz as well once we know the models
        % work
        delete(unzippedMask{1})
        
        % Read trial info
        dataMat = in_bst(sInputs_trials(iInput).FileName);
        
        F = dataMat.F;
        Time = dataMat.Time;
        % The conversion to NIFTI has been done with the sampling rate
        % requested, but downsampling is needed here again for the events
        % creation since I just reloaded the file
        wanted_Fs = sProcess.options.fs.Value{1};
        % And resample if needed
        current_Fs = round(1/diff(Time(1:2)));
        if ~isnan(wanted_Fs) && current_Fs~=wanted_Fs
            %[x, time_out] = process_resample('Compute', x, time_in, NewRate)
            [F, Time] = process_resample('Compute', dataMat.F, dataMat.Time, wanted_Fs);
        end
        
        
        F_segmented = false(size(F));
        
        % Get output study
        [tmp, iStudy] = bst_process('GetOutputStudy', sProcess, sInputs_trials(iInput));
        % Get channel file
        sChannel = bst_get('ChannelForStudy', iStudy);
        % Load channel file
        ChannelMat = in_bst_channel(sChannel.FileName);
        
        % Protocol info
        protocol = bst_get('ProtocolInfo');
        
        % Read the channel pixel coordinates file
        channelCoordinatesFile = bst_fullfile(bst_fileparts(strrep(OutputFiles{iInput}, [protocol.Comment '-segmentation'], [protocol.Comment '-meta-segmentation'])), 'channels.csv');
        T = readtable(channelCoordinatesFile);
        
        for iChannel = 1:length(T.ChannelNames)
            
            [tmp ,indexCh] = ismember(T.ChannelNames{iChannel}, {ChannelMat.Channel.Name});
            
            indicesTime = find(MRI.Cube(T.x_coordinates(iChannel), T.y_coordinates(iChannel), :)==255);
            F_segmented(indexCh,indicesTime) = true;
        end
        
        
        %% Now assign the event if the majority of channels indicate annotation
        % TODO - DETECTION/ANNOTATION ALGORITHM - IMPROVE
        % FOR NOW WORKS ONLY WHEN ALL CHANNELS ARE ANNOTATED - NOT JUST A FEW
        mask = false(1, size(F_segmented,2));
        event_timevector = [];
   
        
        channelsContributingToAnnotation = {};
                              
        for iSample = 1:size(F_segmented,2)
            % If majority - annotate
            if sum(F_segmented(:, iSample))>= length(T.ChannelNames) * sProcess.options.majorityVote.Value{1} / 100
                mask(iSample) = true;
                event_timevector = [event_timevector Time(iSample)];
                annotated_Channels_on_Sample = find(F_segmented(:,iSample));
                
                if strcmp(sProcess.options.annotation.Comment{sProcess.options.annotation.Value}, 'Partial') && ~isempty(annotated_Channels_on_Sample)
                    channelsContributingToAnnotation = unique([channelsContributingToAnnotation  {ChannelMat.Channel(annotated_Channels_on_Sample).Name}]);
                else
                    channelsContributingToAnnotation = [];
                end
            end            
        end
        
        
        % Find discontinuities to assign multiple extended events
        a = diff(mask);
        start = find(a>0) + 1;
        stop = find(a<0);
        
        if length(start)~=length(stop)
            stop % TODO - deal with this
        end
        
%         % Make a summary plot
%         figure(1); 
%         
%         ax = subplot(5,1,[1:4]); imagesc(Time, 1:size(F_segmented,1), F_segmented); ylabel 'Channel ID'; title 'Ivadomed single Trial Segmentation'; set(ax,'Ydir', 'normal')
%         ax2 = subplot(5,1,5); plot(Time, mask); hold on; plot(Time(start), mask(start), '*g', 'linewidth', 8); plot(Time(stop), mask(stop), '*r', 'linewidth', 8); hold off; 
%         title 'Annotation Mask (majority vote)'; xlabel 'Time (sec)'; axis ([min(Time), max(Time), -0.5, 1.5]); yticks([0,1])
%         set(ax,'FontSize',20)
%         ax.XAxis.Visible = 'off'; % remove y-axis
%         set(ax2,'FontSize',20)
        
        %
        detectedEvt = [Time(start);Time(stop)]; % Create extended events
        
        if isRaw
            link_to_raw_file_events_times = [link_to_raw_file_events_times  detectedEvt + slidingWindowTimes(iInput)];
        else
            dataMat = create_new_event(dataMat, detectedEvt, evtName, channelsContributingToAnnotation, sProcess, 0);
            % ===== SAVE RESULT =====
            % Progress bar
            bst_progress('text', 'Saving results...');
    %         bst_progress('set', progressPos + round(3 * iFile / length(sInputs_trials) / 3 * 100));
            % Only save changes if something was detected
            if ~isempty(detectedEvt)
                %dataMat = rmfield(dataMat, 'Time');
                % Save file definition
                bst_save(file_fullpath(sInputs_trials(iInput).FileName), dataMat, 'v6', 1);
                % Report number of detected events
    %             bst_report('Info', sProcess, sInputs_trials(iInput), sprintf('%s: %d events detected in %d categories', chanName, nTotalOcc, nEvents));
            else
                bst_report('Warning', sProcess, sInputs_trials(iInput), ['No event detected.']);
            end
            
        end
    end
    % Return all the input files
    OutputFilesNew = {sInputs_trials.FileName};
    OutputFiles = OutputFilesNew;
    
    if isRaw
        dataMat = in_bst_data(sInputs(1).FileName); % TODO - GENERALIZE TO MULTIPLE LINK TO RAW FILES INPUTS
        dataMat = create_new_event(dataMat, link_to_raw_file_events_times, evtName, channelsContributingToAnnotation, sProcess, 1);
        
        ProtocolInfo = bst_get('ProtocolInfo');
        bst_save(bst_fullfile(ProtocolInfo.STUDIES, sInputs(1).FileName), dataMat, 'v6');
        
        %% Delete the created trials
    
        % %     % Delete trials - TODO
    
        OutputFiles = {sInputs(1).FileName};
    end
end

function dataMat = create_new_event(dataMat, detectedEvt, evtName, channelsContributingToAnnotation, sProcess, isRaw)
% ===== CREATE EVENTS =====
    if isRaw
        events = dataMat.F.events;
    else
        events = dataMat.Events;
    end

    sEvent = [];
    % Basic events structure
    if isempty(events)
        events = repmat(db_template('event'), 0);
    end
    % Get the event to create
    iEvt = find(strcmpi({events.label}, evtName));
    % Existing event: reset it
    if ~isempty(iEvt)
        sEvent = events(iEvt);
        sEvent.epochs     = [];
        sEvent.times      = [];
        sEvent.reactTimes = [];
    % Else: create new event
    else
        % Initialize new event
        iEvt = length(events) + 1;
        sEvent = db_template('event');
        sEvent.label = evtName;
        % Get the default color for this new event
        sEvent.color = panel_record('GetNewEventColor', iEvt, events);
    end
    % Times, epochs
    if ~isempty(sProcess.options.timewindow_annot.Value{1})
        sEvent.times  = detectedEvt - (mean(sProcess.options.timewindow_annot.Value{1})); % Centers the prediction to the center of the annotation
    else
        sEvent.times  = detectedEvt;
    end
    sEvent.epochs = ones(1, size(sEvent.times,2));

    if strcmp(sProcess.options.annotation.Comment{sProcess.options.annotation.Value}, 'Partial')
        sEvent.channels = cell(1, size(sEvent.times, 2));
        for iEvent = 1:size(sEvent.times, 2)
            sEvent.channels{iEvent} = channelsContributingToAnnotation;
        end
    else
        sEvent.channels = cell(1, size(sEvent.times, 2));
    end
    sEvent.notes    = cell(1, size(sEvent.times, 2));
    % Add to events structure    
    if isRaw
        dataMat.F.events(iEvt) = sEvent;
    else
        dataMat.Events(iEvt) = sEvent;
    end
    
end


