function varargout = process_ivadomed( varargin )
% PROCESS_IVADOMED: this function enables training of deep learning models 
% with the Ivadomed toolbox:
% https://ivadomed.org/en/latest/index.html

% USAGE:    sProcess = process_ivadomed('GetDescription')
%        OutputFiles = process_ivadomed('Run', sProcess, sInput)

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
    sProcess.Comment     = 'Ivadomed';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3112;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'data'};
    sProcess.OutputTypes = {'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    
    
    sProcess.options.label1.Comment = '<B>BIDS conversion parameters:</B>';
    sProcess.options.label1.Type    = 'label';
    % Event name
    sProcess.options.eventname.Comment = 'Event for ground truth';
    sProcess.options.eventname.Type    = 'text';
    sProcess.options.eventname.Value   = 'event1';
    % Event help comment
    sProcess.options.eventname_help.Comment = '<I><FONT color="#777777">If the eventname is left empty, the annotations is based on the following time-window within each trial</FONT></I>';
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
    sProcess.options.jitter.Value   = {100, 'ms', 0};
    % Jitter comment
    sProcess.options.jitter_help.Comment = '<I><FONT color="#777777">This is used to crop the edges of each trial so the trained model doesn"t learn the position of the event</FONT></I>';
    sProcess.options.jitter_help.Type    = 'label';
    
    % Needed threshold for soft annotation
    sProcess.options.annotthresh.Comment = 'Soft annotation threshold (0,1)';
    sProcess.options.annotthresh.Type    = 'value';
    sProcess.options.annotthresh.Value   = {[], [], 1};
    % Annotation threshold comment
    sProcess.options.annotthresh_help.Comment = '<I><FONT color="#777777">If selected, the annotation will have a soft threshold. Leave empty for hard annotation at 0.5</FONT></I>';
    sProcess.options.annotthresh_help.Type    = 'label';
    % Parallel processing
    sProcess.options.paral.Comment = 'Parallel processing';
    sProcess.options.paral.Type    = 'checkbox';
    sProcess.options.paral.Value   = 0;
    % Method: BIDS subject selection
    sProcess.options.label2.Comment = '<B>BIDS folders creation </B>';
    sProcess.options.label2.Type    = 'label';
    sProcess.options.bidsFolders.Comment = {'Normal', 'Separate runs/sessions as different subjects'};
    sProcess.options.bidsFolders.Type    = 'radio';
    sProcess.options.bidsFolders.Value   = 1;
    
    % Method: Command to use
    sProcess.options.label3.Comment = '<BR><B>Command to execute:</B>';
    sProcess.options.label3.Type    = 'label';
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
     % Method: Average or PCA
    sProcess.options.label4.Comment = '<B>Model selection:</B>';
    sProcess.options.label4.Type    = 'label';
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
    sProcess.options.label5.Comment = '<B>Slice Axis:</B>';
    sProcess.options.label5.Type    = 'label';
    sProcess.options.sliceaxis.Comment = {'Axial'; 'Sagittal'; 'Coronal'};
    sProcess.options.sliceaxis.Type    = 'radio';
    sProcess.options.sliceaxis.Value   = 1;
    % Loss function name
    sProcess.options.loss.Comment = '<B>Loss function:</B>';
    sProcess.options.loss.Type    = 'text';
    sProcess.options.loss.Value   = 'DiceLoss';
    % Uncertainty
    sProcess.options.label6.Comment = '<B>Uncertainty</B>';
    sProcess.options.label6.Type    = 'label';
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
    if isfield(sProcess.options, 'modelselection') && ~isempty(sProcess.options.modelselection.Value)
        Comment = ['Ivadomed: ' sProcess.options.modelselection.Comment{sProcess.options.modelselection.Value}];
    else
        Comment = 'Ivadomed';
    end
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs)
    global GlobalData

    % TODO - Confirm this is the best approach to check if IVADOMED is
    % installed
    % Check if ivadomed is installed on the Conda environment
    output = system('ivadomed -h');
    if output~=0
        bst_report('Error', sProcess, sInputs, 'Ivadomed package is not accessible. Are you running Matlab through an anaconda environment that has Ivadomed installed?');
        return
    end
    
    wanted_Fs = sProcess.options.fs.Value{1};
    
    
    %% Do some checks on the parameters
    
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
            [isSelectedEventPresent, index] = ismember(sProcess.options.eventname.Value, {events.label});
            
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
    
    
    %% Gather F for all files here - This shouldn't create a memory issue
    % Reason why all of them are imported here is that in_bst doesn't like
    % to be parallelized (as far as I checked), so can't call it within 
    % convertTopography2matrix
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                       'IvadomedNiftiFiles', ...
                       protocol.Comment);
    
    info_trials = struct;
    
    for iInput = 1:length(sInputs)
        info_trials(iInput).FileName = sInputs(iInput).FileName;
        info_trials(iInput).subject = lower(str_remove_spec_chars(sInputs(iInput).SubjectName));
        info_trials(iInput).session = lower(str_remove_spec_chars(sInputs(iInput).Condition));
        info_trials(iInput).trial = lower(str_remove_spec_chars(sInputs(iInput).Comment));  % TODO - THE UNIQUE LIST OF TRIAL LABELS NEEDS TO BE ADDED ON THE IVADOMED CONFIG.JSON FILE TO BE INCLUDED IN TRAINING/VALIDATION
      
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
    
    if isempty(poolobj)
        bst_progress('start', 'Ivadomed', 'Converting trials to NIFTI files...', 0, length(sInputs));
        for iFile = 1:length(info_trials)
            sInput = sInputs(iFile);
            [filenames{iFile}, subjects{iFile}] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
            bst_progress('inc', 1);
        end
    else
        bst_progress('start', 'Raster Plot per Neuron', 'Binning Spikes...', 0, 0);
        parfor iFile = 1:length(info_trials)
            sInput = sInputs(iFile);
            [filenames{iFile}, subjects{iFile}] = convertTopography2matrix(info_trials(iFile), sProcess, iFile, figures_struct);
        end
    end
        
    OutputFiles = {};
    
    % === EXPORT BIDS FILES ===
    export_participants_tsv(parentPath, unique(subjects))
    export_participants_json(parentPath)
    export_dataset_description(parentPath)
    export_readme(parentPath)
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
    
    %% ADD A JITTER
    % We want to avoid the model learning the positioning of the event so
    % we crop the time dimension on both sides with a jitter
    
    current_Fs = round(1/diff(single_info_trial.dataMat.Time(1:2)));
    
    discardElementsBeginning = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);
    discardElementsEnd = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);
    
    single_info_trial.dataMat.Time = single_info_trial.dataMat.Time(1+discardElementsBeginning:end-discardElementsEnd);
    single_info_trial.dataMat.F = single_info_trial.dataMat.F(:,1+discardElementsBeginning:end-discardElementsEnd);
    
    
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
       if strcmp(single_info_trial.ChannelMat.Channel(iChannel).Type, 'EEG') || strcmp(single_info_trial.ChannelMat.Channel(iChannel).Type, 'MEG')  %% ACCOMMODATE MORE HERE - fNIRS?
          nElectrodes = nElectrodes + 1;
          selectedChannels(end + 1) = iChannel;
       end
    end
    
    %% Gather the topography slices to a single 3d matrix
    % Here the time dimension is the 3rd dimension
    figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct);
    NIFTI = channelMatrix2pixelMatrix(single_info_trial.dataMat.F, single_info_trial.dataMat.Time, selectedChannels, iFile, figures_struct);
%     figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct);

    %% Get the output filename
    
    % Substitute the voxels with the 2D slices created from the 2dlayout
    % topography
    single_info_trial.sMri.Cube = NIFTI;
    
    % Get output filename
    if sProcess.options.bidsFolders.Value==1
        OutputMriFile = bst_fullfile(single_info_trial.parentPath, ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial '.nii']);
    elseif sProcess.options.bidsFolders.Value==2
        subject = [subject session];
        OutputMriFile = bst_fullfile(single_info_trial.parentPath, ['sub-' subject], 'anat', ['sub-' subject '_' trial '.nii']);
    end
    
    
    %% Export the created cube to NIFTI
    OutputMriFile = export2NIFTI(single_info_trial.sMri, OutputMriFile);
    
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
    
    F_derivative = ones(size(single_info_trial.dataMat.F));    
                     
    if isempty(single_info_trial.dataMat.Events)
        iAllSelectedEvents = [];
    else
        iAllSelectedEvents = find(ismember({single_info_trial.dataMat.Events.label}, strsplit(sProcess.options.eventname.Value,{',',' '})));
    end
    annotationValue = 0;
        
    if ~isempty(iAllSelectedEvents)  % Selected event
        for iSelectedEvent = iAllSelectedEvents
            annotationValue = annotationValue-1;
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
                    end
                end
            end
        end
    else  % No event selected - ANNOTATE BASED ON THE SELECTED TIME WINDOW WITHIN THE TIME IN TRIAL
    	annotationValue = annotationValue-1;
        iAnnotation_time_edges  = bst_closest(sProcess.options.timewindow.Value{1}, single_info_trial.dataMat.Time);
        % Annotate the entire slice
        F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
    end
        
    figures_struct = open_close_topography_window(single_info_trial.FileName, 'open', iFile, figures_struct);
    NIFTI_derivative = channelMatrix2pixelMatrix(F_derivative, single_info_trial.dataMat.Time, selectedChannels, iFile, figures_struct);
    figures_struct = open_close_topography_window(single_info_trial.FileName, 'close', iFile, figures_struct);

    
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
    
    % TODO - IF MULTIPLE SUBJECTS OR MULTIPLE SESSIONS - ACCOMMODATE THE MAIN
    % FOLDER STRUCTURE
    if isempty(sProcess.options.eventname.Value)
        annotation = 'centered';
    else
        annotation = sProcess.options.eventname.Value;
    end
    
    % Get output filename
    if sProcess.options.bidsFolders.Value==1
        OutputDerivativeMriFile = bst_fullfile(single_info_trial.parentPath, 'derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_' trial '_' annotation '.nii']);
    elseif sProcess.options.bidsFolders.Value==2
        % Subject already has the session integrated
        OutputDerivativeMriFile = bst_fullfile(single_info_trial.parentPath, 'derivatives', 'labels', ['sub-' subject], 'anat', ['sub-' subject '_' trial '_' annotation '.nii']);
    end
    
    %% Export the created cube to NIFTI
    OutputMriFile = export2NIFTI(single_info_trial.sMri, OutputDerivativeMriFile);
    
end

function figures_struct = open_close_topography_window(FileName, action, iFile, figures_struct)
    global GlobalData
    if strcmp(action, 'open')
        %% Open a window to inherit properties
        %[hFig, iDS, iFig] = view_topography(DataFile, Modality, TopoType, F)
        % TODO - consider adding flag on view_topography for not displaying the
        % figure when it is for Ivadomed
        % Modality       : {'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'ECOG', 'SEEG', 'NIRS'}
        % TopoType       : {'3DSensorCap', '2DDisc', '2DSensorCap', 2DLayout', '3DElectrodes', '3DElectrodes-Cortex', '3DElectrodes-Head', '3DElectrodes-MRI', '3DOptodes', '2DElectrodes'}
        
        
            % TODO - GET MODALITY AUTOMATICALLY
        [hFig, iDS, iFig] = view_topography(FileName, 'MEG', '2DSensorCap');        
%         [hFig, iFig, iDS] = bst_figures('GetFigure', GlobalData.DataSet(iFile).Figure.hFigure);
%         bst_figures('SetBackgroundColor', hFig, [1 1 1]);




        set(hFig, 'Visible', 'off');





%         set(hFig, 'Position', [left bottom width height]);
        set(hFig, 'Position', [0 0 710 556]);  % Default values of single 2dlayout figure
        
        % Find index that just opened figure corresponds to (this is done for enabling parallelization)
        all_datafiles = {GlobalData.DataSet.DataFile};
        [temp, index] = ismember(FileName, all_datafiles);
        
        figures_struct(iFile).FigureObject = GlobalData.DataSet(index).Figure;
        figures_struct(iFile).Status = 'Open';
        
%         % If the colorbar object is present delete it
%         if length(figures_struct(iFile).FigureObject.hFigure.Children) > 1
%            delete(figures_struct(iFile).FigureObject.hFigure.Children(1))
%         end
        
%              [hFigs,iFigs,iDSs,iSurfs] = bst_figures('DeleteFigure', hFig, 'NoUnload')



        %TODO - MAKE SURE YOU SET THE COLORMAP TO BE GRAY BEFORE SAVING THE
        %SLICES



    elseif strcmp(action, 'close')
        % Close window
%         close(GlobalData.DataSet(iFile).Figure.hFigure)
        close(figures_struct(iFile).FigureObject.hFigure)
        figures_struct(iFile).Status = 'Closed';
    end
end


function NIFTI = channelMatrix2pixelMatrix(F, Time, selectedChannels, iFile, figures_struct)
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
    
%             plot(figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1), figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2),'bo')
%             delete(figures_struct(iFile).FigureObject.hFigure.Children(1)) % Gets rid of the colorbar object
%     
    
    % Get size of exported files
%     [height,width,~] = size(print(GlobalData.DataSet(iFile).Figure.hFigure, '-noui', '-r50', '-RGBImage'));
    [height,width,~] = size(print(figures_struct(iFile).FigureObject.hFigure, '-noui', '-r50', '-RGBImage'));

    
   
    
%         axes_handle = figures_struct(iFile).FigureObject.hFigure.CurrentAxes;
%         set(axes_handle,'units','pixels');
%         pos = get(axes_handle,'position');
%         xlim = get(axes_handle,'xlim');
%         ylim = get(axes_handle,'ylim');
% %         x_in_pixels = pos(1) + pos(3) * (x_in_axes-xlim(1))/(xlim(2)-xlim(1));
% 
%             hardcoded_offset = 80
% 
%         x_in_pixels = pos(1) + pos(3) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,1)-xlim(1))/(xlim(2)-xlim(1));
%         y_in_pixels = hardcoded_offset + pos(2) + pos(4) * (figures_struct(iFile).FigureObject.Handles.MarkersLocs(:,2)-ylim(1))/(ylim(2)-ylim(1));
    
              
    NIFTI = zeros(height, width, length(Time), 'uint8');
    for iTime = 1:length(Time)
        DataToPlot = F(selectedChannels,iTime);

        % ===== APPLY TRANSFORMATION =====
        % Mapping on a different surface (magnetic source reconstruction of just smooth display)
%         if ~isempty(GlobalData.DataSet(iFile).Figure.Handles.Wmat)
        if ~isempty(figures_struct(iFile).FigureObject.Handles.Wmat)
            % Apply interpolation matrix sensors => display surface
%             if (size(GlobalData.DataSet(iFile).Figure.Handles.Wmat,1) == length(DataToPlot))
            if (size(figures_struct(iFile).FigureObject.Handles.Wmat,1) == length(DataToPlot))
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
%         img = print(GlobalData.DataSet(iFile).Figure.hFigure, '-noui', '-r50', '-RGBImage');        
        img = print(figures_struct(iFile).FigureObject.hFigure, '-noui', '-r50', '-RGBImage');        
        img_gray= 255 - rgb2gray(img);
        NIFTI(:,:,iTime) = img_gray;
        
        
    end
    
    
    %% CROP IMAGE (AVOID COLORBAR AND WASTED SPACE AROUND TOPOGRAPHY)
    % TODO - THIS IS AFFECTED BY THE SCREEN RESOLUTION
    % ALSO FLIP TO CREATE CORRECT ORIENTATION WITH
    % LEFT-RIGHT-ANTERIOR-POSTERIOR
       % USE THIS FOR -r0
%     crop_from_top = 60;
%     crop_from_bottom = -70;
%     crop_from_left = 70;
%     crop_from_right = -140;
    
    % USE THIS FOR -r50
    crop_from_top = 35;
    crop_from_bottom = -35;
    crop_from_left = 40;
    crop_from_right = -75;
    
    NIFTI = NIFTI(crop_from_top:end+crop_from_bottom, crop_from_left:end+crop_from_right,:);    
    NIFTI = flip(permute(NIFTI,[2,1,3]),2);
    
    
    
    %% CONSIDER DOWNSAMPLING - THIS WILL AFFECT THE COORDINATES OF THE ELECTRODES - TODO
    
% %     NIFTI = imresize(NIFTI, [200, 150], 'nearest');
%     
%     figure(1);
%     imagesc(squeeze(NIFTI2(:,:,50)));

    
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




