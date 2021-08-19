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
    % Conversion of both trials and their derivatives
    sProcess.options.convert.Type   = 'text';
    sProcess.options.convert.Value  = 'segmentation';  % Other option: 'conversion'
    sProcess.options.convert.Hidden = 1;
    % BIDS subject selection
    sProcess.options.bidsFolders.Comment = {'Normal', 'Separate runs/sessions as different subjects'};
    sProcess.options.bidsFolders.Type    = 'radio';
    sProcess.options.bidsFolders.Value   = 1;
    sProcess.options.bidsFolders.Hidden = 1;
    % Parallel processing
    sProcess.options.paral.Comment = 'Parallel processing';
    sProcess.options.paral.Type    = 'checkbox';
    sProcess.options.paral.Value   = 0;
    sProcess.options.paral.Hidden = 1;
    
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = ['Detect: ', sProcess.options.eventname.Value];
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs) %#ok<DEFNU>   
    % ===== GET OPTIONS =====
    % Event name
    evtName = strtrim(sProcess.options.eventname.Value);
    modelFile = sProcess.options.evtfile.Value{1};
    
    if isempty(evtName) || isempty(modelFile)
        bst_report('Error', sProcess, [], 'Event name and trained model must be specified.');
        OutputFiles = {};
        return;
    end
    
    %% OPTION TO CHANGE THE CONFIG FILE
    %  PROBABLY WONT BE NEEDED - FLAG CALLS SHOULD BE ENOUGH FOR
    %  SEGMENTATION
    
    % Grab the config.json file that was used and assign the method to
    % perform segmentation
    
%     ivadomedOutputFolder = bst_fileparts(bst_fileparts(modelFile));
%     configFile = bst_fullfile(ivadomedOutputFolder, 'config_file.json');
%     
%     fid = fopen(configFile);
%     raw = fread(fid,inf);
%     str = char(raw');
%     fclose(fid);
%     config_struct = jsondecode(str);
%     
%     config_struct.command = 'segment';
%     
%     % Save back to json (Make it a bit more readable - still not great)
%     txt = jsonencode(config_struct);
%     txt = strrep(txt, ',', sprintf(',\r'));
%     txt = strrep(txt, '[{', sprintf('[\r{\r'));
%     txt = strrep(txt, '}]', sprintf('\r}\r]'));
%     
%     fid = fopen(configFile, 'w');
%     fwrite(fid, txt);
%     fclose(fid);

    %% Check if IVADOMED is installed/accessible
    
    output = system('ivadomed -h');
    if output~=0
        bst_report('Error', sProcess, sInputs, 'Ivadomed package is not accessible. Are you running Matlab through an anaconda environment that has Ivadomed installed?');
        return
    end
    
    
    
    %% Important files/folders
    ivadomedOutputFolder = bst_fileparts(bst_fileparts(modelFile));  % Output of the trained model
    configFile = bst_fullfile(ivadomedOutputFolder, 'config_file.json');
    
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '-segmentation']);
    channelsparentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       [protocol.Comment '-meta-segmentation']);
                   
    %% Create a BIDS dataset with the trials to be segmented
    get_filenames = 1;
    OutputFiles = process_ivadomed('Run', sProcess, sInputs, get_filenames);    
    
    
    %% Instead of changing the config.json file, call ivadomed with FLAG usage
    % to perform segmentation of the selected files 
    
    output = system(['ivadomed --segment -c ' configFile ' -pd ' parentPath ' -po ' ivadomedOutputFolder]);
    if output~=0
        bst_report('Error', sProcess, sInputs, 'Something went wrong during segmentation');
        return
    end
    
    %% Get ivadomed output segmentation files
    segmentationMasks = cell(length(sInputs),1);
    for iInput = 1:length(sInputs)
        [a,b,c] = bst_fileparts(OutputFiles{iInput});
        [a,file_basename,c] = bst_fileparts(b);

        % TODO - GRAB event annotation suffix (e.g. "centered") from json file
        segmentationMasks{iInput} = bst_fullfile(ivadomedOutputFolder, 'pred_masks', [file_basename, '_', 'centered', '_pred.nii.gz']);
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
        F = false(size(dataMat.F));
        Time = dataMat.Time;
        
        % Get output study
        [tmp, iStudy] = bst_process('GetOutputStudy', sProcess, sInputs(iInput));
        % Get channel file
        sChannel = bst_get('ChannelForStudy', iStudy);
        % Load channel file
        info_channels(iInput).ChannelMat = in_bst_channel(sChannel.FileName);
        
        % Read the channel pixel coordinates file
        channelCoordinatesFile = bst_fullfile(bst_fileparts(strrep(OutputFiles{iInput}, 'brainstorm-segmentation', 'brainstorm-meta-segmentation')), 'channels.csv');
        T = readtable(channelCoordinatesFile);
        
        for iChannel = 1:length(T.ChannelNames)
            
            [tmp ,indexCh] = ismember(T.ChannelNames{iChannel}, {info_channels(iInput).ChannelMat.Channel.Name});
            
            indicesTime = find(MRI.Cube(T.x_coordinates(iChannel), T.y_coordinates(iChannel), :)==255);
            F(indexCh,indicesTime) = true;
        end
        
        
        %% Now assign the event if the majority of channels indicate annotation
        % TODO - DETECTION/ANNOTATION ALGORITHM - IMPROVE
        % FOR NOW WORKS ONLY WHEN ALL CHANNELS ARE ANNOTATED - NOT JUST A FEW
        mask = false(1, size(F,2));
        event_timevector = [];
        
        
        majority_vote = 0.8;  % this allows to allocate the percentage of channels that need to have the annotation in order to keep it
                              % This is useful only in the case where all
                              % of annotations on all channels - not
                              % partial annotations - TODO - generalize to
                              % partial annotations
        
        for iSample = 1:size(F,2)
            % If majority - annotate
            if sum(F(:, iSample))>= length(T.ChannelNames) * majority_vote
                mask(iSample) = true;
                event_timevector = [event_timevector Time(iSample)];
            end            
        end
        
        
        % Find discontinuities to assign multiple extended events
        a = diff(mask);
        start = find(a>0);
        stop = find(a<0);
        
        if length(start)~=length(stop)
            stop % deal with this
        end
        
%         % Make a summary plot
%         figure(1); 
%         
%         ax = subplot(5,1,[1:4]); imagesc(Time, 1:size(F,1), F); ylabel 'Channel ID'; title 'Ivadomed single Trial Segmentation'; set(ax,'Ydir', 'normal')
%         ax2 = subplot(5,1,5); plot(Time, mask); hold on; plot(Time(start+1), mask(start+1), '*g', 'linewidth', 8); plot(Time(stop), mask(stop), '*r', 'linewidth', 8); hold off; 
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
            % Event name
            if (i > 1)
                newName = sprintf('%s%d', evtName, i);
            else
                newName = evtName;
            end
            % Get the event to create
            iEvt = find(strcmpi({dataMat.Events.label}, newName));
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
                sEvent.label = newName;
                % Get the default color for this new event
                sEvent.color = panel_record('GetNewEventColor', iEvt, dataMat.Events);
            end
            % Times, samples, epochs
            sEvent.times    = detectedEvt;
            sEvent.epochs   = ones(1, size(sEvent.times,2));
            sEvent.channels = cell(1, size(sEvent.times, 2));
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
     