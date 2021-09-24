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
% Authors: Konstantinos Nasiotis 2021

eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    sProcess.Comment     = 'Ivadomed Segmentation';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3113;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'raw', 'data'};
    sProcess.OutputTypes = {'raw', 'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    % Event name
    sProcess.options.eventname.Comment = 'Event name: ';
    sProcess.options.eventname.Type    = 'text';
    sProcess.options.eventname.Value   = 'segmented';
    % Model selection options
    SelectOptions = {...
        '', ...                               % Filename
        '', ...                               % FileFormat
        'open', ...                           % Dialog type: {open,save}
        'Import events...', ...               % Window title
        'ImportData', ...                     % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                         % Selection mode: {single,multiple}
        'files', ...                          % Selection mode: {files,dirs,files_and_dirs}
        bst_get('FileFilters', 'model'), ... % Get all the available file formats
        'EventsIn'};                          % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn
    % Option: Event file
    sProcess.options.evtfile.Comment = 'Deep learning model file:';
    sProcess.options.evtfile.Type    = 'filename';
    sProcess.options.evtfile.Value   = SelectOptions;
    % Needed Fs
    sProcess.options.fs.Comment = 'Sampling rate that was used  while training the model <I><FONT color="#FF0000">(AUTOMATE THIS)</FONT></I>';
    sProcess.options.fs.Type    = 'value';
    sProcess.options.fs.Value   = {100, 'Hz', 0};
    % GPU ID to run inference on
    sProcess.options.gpu.Comment = 'GPU ID to run inference on <I><FONT color="#FF0000">(AUTOMATE THIS)</FONT></I>';
    sProcess.options.gpu.Type    = 'value';
    sProcess.options.gpu.Value   = {0, [], 0};
    % Conversion of both trials and their derivatives
    sProcess.options.convert.Type   = 'text';
    sProcess.options.convert.Value  = 'segmentation';  % Other option: 'conversion'
    sProcess.options.convert.Hidden = 1;
    % BIDS subject selection
    % Method: BIDS subject selection
    sProcess.options.annotationLabel.Comment = '<I><FONT color="#FF0000">Whole Head annotation (all channels) or partial (annotate a few channels)</FONT></I>';
    sProcess.options.annotationLabel.Type    = 'label';
    sProcess.options.annotation.Comment = {'Whole', 'Partial'};
    sProcess.options.annotation.Type    = 'radio';
    sProcess.options.annotation.Value   = 1;
    % BIDS subject selection
    sProcess.options.bidsFolders.Comment = {'Normal', 'Separate runs/sessions as different subjects'};
    sProcess.options.bidsFolders.Type    = 'radio';
    sProcess.options.bidsFolders.Value   = 1;
    sProcess.options.bidsFolders.Hidden = 1;
    % GPU ID to run inference on - % this allows to allocate the percentage
    % of channels that need to have the annotation in order to keep the
    % annotation
    sProcess.options.majorityVote.Comment = 'Majority Vote Percentage [0,100] <I><FONT color="#FF0000">OF TOTAL MEG/EEG CHANNELS</FONT></I>';
    sProcess.options.majorityVote.Type    = 'value';
    sProcess.options.majorityVote.Value   = {50, [], 0};
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
    
    
    
    
    %% CHECK IVADOMED EXISTS
    fname = bst_fullfile(bst_get('UserPluginsDir'), 'ivadomed', 'ivadomed-master', 'ivadomed', 'main.py');
    if ~(exist(fname, 'file') == 2)
        
        % Check if ivadomed can be accessed from a system call
        % in case the user installed it outside of Brainstorm
        output = system('ivadomed -h');
        if output~=0
            bst_report('Error', sProcess, sInputs, 'Ivadomed package is not accessible. Are you running Matlab through an anaconda environment that has Ivadomed installed?');
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
    modelFile = sProcess.options.evtfile.Value{1};
    
    if isempty(evtName) || isempty(modelFile)
        bst_report('Error', sProcess, [], 'Event name and trained model must be specified.');
        OutputFiles = {};
        return;
    end
    
    %% CHANGE THE CONFIG FILE TO RUN LOCALLY
    
    % Grab the config.json file that was used and assign the gpu that the
    % user selected
    
    ivadomedOutputFolder = bst_fileparts(bst_fileparts(modelFile));
    configFile = bst_fullfile(ivadomedOutputFolder, 'config_file.json');
    
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
    txt = jsonencode(config_struct, 'PrettyPrint', true);
    
    fid = fopen(configFile, 'w');
    fwrite(fid, txt);
    fclose(fid);

    
    %% Important files/folders
    ivadomedOutputFolder = bst_fileparts(bst_fileparts(modelFile));  % Output of the trained model
    
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '-segmentation']);
    channelsparentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '-meta-segmentation']);
                   
    %% Create a BIDS dataset with the trials to be segmented
    get_filenames = 1;
    OutputFiles = process_ivadomed_create_dataset('Run', sProcess, sInputs, get_filenames);    
    
    
    %% Call ivadomed with "segment" method
    output = system([ivadomed_call ' --segment -c ' configFile ' -pd ' parentPath ' -po ' ivadomedOutputFolder]);
    if output~=0
        bst_report('Error', sProcess, sInputs, 'Something went wrong during segmentation');
        return
    end
    
    %% Get ivadomed output segmentation files
    segmentationMasks = cell(length(sInputs),1);
    for iInput = 1:length(sInputs)
        [a,b,c] = bst_fileparts(OutputFiles{iInput});
        [a,file_basename,c] = bst_fileparts(b);

        % Grab event annotation suffix (e.g. "centered") from json file
        segmentationMasks{iInput} = bst_fullfile(ivadomedOutputFolder, 'pred_masks', [file_basename, config_struct.loader_parameters.target_suffix{1}, '_pred.nii.gz']);
    end
    
    %% Converter from masks to Brainstorm events
    nEvents = 0;
    nTotalOcc = 0;
    
    results = struct;
    
    for iInput = 1:length(sInputs)
        MriFile = segmentationMasks{iInput};
        unzippedMask = gunzip(MriFile);

        [MRI, vox2ras] = in_mri_nii(unzippedMask{1}, 1, 1, 0);

        % Delete nifti (.nii) - keep the unzipped for debugging - TODO -
        % consider removing the .nii.gz as well once we know the models
        % work
        delete(unzippedMask{1})
        
        % Read trial info
        dataMat = in_bst(sInputs(iInput).FileName);
        
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
        [tmp, iStudy] = bst_process('GetOutputStudy', sProcess, sInputs(iInput));
        % Get channel file
        sChannel = bst_get('ChannelForStudy', iStudy);
        % Load channel file
        ChannelMat = in_bst_channel(sChannel.FileName);
        
        % Read the channel pixel coordinates file
        channelCoordinatesFile = bst_fullfile(bst_fileparts(strrep(OutputFiles{iInput}, 'brainstorm-segmentation', 'brainstorm-meta-segmentation')), 'channels.csv');
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
                
                if strcmp(sProcess.options.annotation.Comment{1}, 'Partial')
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
            stop % deal with this
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
        

        % ===== CREATE EVENTS =====
        sEvent = [];
        % Basic events structure
        if isempty(dataMat.Events)
            dataMat.Events = repmat(db_template('event'), 0);
        end
        % Process each event type separately
        for i = 1:size(detectedEvt,2)
            % Get the event to create
            iEvt = find(strcmpi({dataMat.Events.label}, evtName));
            % Existing event: reset it
            if ~isempty(iEvt)
                sEvent = dataMat.Events(iEvt);
                sEvent.epochs     = [];
                sEvent.times      = [];
                sEvent.reactTimes = [];
            % Else: create new event
            else
                % Initialize new event
                iEvt = length(dataMat.Events) + 1;
                sEvent = db_template('event');
                sEvent.label = evtName;
                % Get the default color for this new event
                sEvent.color = panel_record('GetNewEventColor', iEvt, dataMat.Events);
            end
            % Times, samples, epochs
            sEvent.times    = detectedEvt;
            sEvent.epochs   = ones(1, size(sEvent.times,2));
            
            if strcmp(sProcess.options.annotation.Comment{1}, 'Partial')
                sEvent.channels = cell(1, size(sEvent.times, 2));
                for iEvent = 1:size(sEvent.times, 2)
                    sEvent.channels{iEvent} = channelsContributingToAnnotation;
                end
            else
                sEvent.channels = cell(1, size(sEvent.times, 2));
            end
            sEvent.notes    = cell(1, size(sEvent.times, 2));
            % Add to events structure
            dataMat.Events(iEvt) = sEvent;
            nEvents = nEvents + 1;
            nTotalOcc = nTotalOcc + size(sEvent.times, 2);
        end

        % ===== SAVE RESULT =====
        % Progress bar
        bst_progress('text', 'Saving results...');
%         bst_progress('set', progressPos + round(3 * iFile / length(sInputs) / 3 * 100));
        % Only save changes if something was detected
        if ~isempty(sEvent)
            %dataMat = rmfield(dataMat, 'Time');
            % Save file definition
            bst_save(file_fullpath(sInputs(iInput).FileName), dataMat, 'v6', 1);
            % Report number of detected events
%             bst_report('Info', sProcess, sInputs(iInput), sprintf('%s: %d events detected in %d categories', chanName, nTotalOcc, nEvents));
        else
            bst_report('Warning', sProcess, sInputs(iInput), ['No event detected. Please check the annotations quality.']);
        end
    end
    % Return all the input files
    OutputFiles = {sInputs.FileName};
        
        
end
     