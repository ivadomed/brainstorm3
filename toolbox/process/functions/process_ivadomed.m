function varargout = process_ivadomed( varargin )

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
    
    % Event name
    sProcess.options.eventname.Comment = 'Event for ground truth';
    sProcess.options.eventname.Type    = 'text';
    sProcess.options.eventname.Value   = 'event1';
    
    % Needed Fs
    sProcess.options.fs.Comment = 'Resampling rate (empty for no resampling)';
    sProcess.options.fs.Type    = 'value';
    sProcess.options.fs.Value   = {100, 'Hz', 0};
    
    % Needed Jitter for cropping
    sProcess.options.jitter.Comment = 'Jitter value';
    sProcess.options.jitter.Type    = 'value';
    sProcess.options.jitter.Value   = {100, 'ms', 0};
    % Jitter comment
    sProcess.options.jitter_help.Comment = '<I><FONT color="#777777">This is used to crop the edges of each trial so the trained model doesn"t learn the position of the event</FONT></I>';
    sProcess.options.jitter_help.Type    = 'label';
    
    
    % Method: Average or PCA
    sProcess.options.label1.Comment = '<BR>Command to execute:';
    sProcess.options.label1.Type    = 'label';
    sProcess.options.command.Comment = {'Training', 'Testing', 'Segmentation'};
    sProcess.options.command.Type    = 'radio';
    sProcess.options.command.Value   = 1;
    
    % File selection options
    SelectOptions = {...
        '/tmp/spinegeneric', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'Import anatomy folder...', ...    % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                      % Selection mode: {single,multiple}
        'dirs', ...                        % Selection mode: {files,dirs,files_and_dirs}
        bst_get('FileFilters', 'AnatIn'), ... % Available file formats
        'AnatIn'};                         % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    
    % Use existing SSPs
    sProcess.options.usessp.Comment = 'Debugging';
    sProcess.options.usessp.Type    = 'checkbox';
    sProcess.options.usessp.Value   = 1;
    % Parallel processing
    sProcess.options.paral.Comment = 'Parallel processing';
    sProcess.options.paral.Type    = 'checkbox';
    sProcess.options.paral.Value   = 0;
    % Event name
    sProcess.options.gpu.Comment = 'GPU IDs: ';
    sProcess.options.gpu.Type    = 'text';
    sProcess.options.gpu.Value   = '1, 2, 3';
    % Option: Dataset Selection
    sProcess.options.output.Comment = 'Output Folder:';
    sProcess.options.output.Type    = 'filename';
    sProcess.options.output.Value   = SelectOptions;
    
    
    % Default selection of components
    sProcess.options.gpu.Comment = 'GPU IDs: ';
    sProcess.options.gpu.Type    = 'value';
    sProcess.options.gpu.Value   = {[0,1,2,3], 'list', 0};
     % Method: Average or PCA
    sProcess.options.label3.Comment = '<BR>Model selection:';
    sProcess.options.label3.Type    = 'label';
    sProcess.options.modelselection.Comment = {'default_model'; 'FiLMedUnet'; 'HeMISUnet'; 'Modified3DUNet'};
    sProcess.options.modelselection.Type    = 'radio';
    sProcess.options.modelselection.Value   = 1;
    
    % File selection options
    SelectOptions = {...
        '/data/large-dataset-testing', ...                            % Filename
        '', ...                            % FileFormat
        'open', ...                        % Dialog type: {open,save}
        'Import anatomy folder...', ...    % Window title
        'ImportAnat', ...                  % LastUsedDir: {ImportData,ImportChannel,ImportAnat,ExportChannel,ExportData,ExportAnat,ExportProtocol,ExportImage,ExportScript}
        'single', ...                      % Selection mode: {single,multiple}
        'dirs', ...                        % Selection mode: {files,dirs,files_and_dirs}
        bst_get('FileFilters', 'AnatIn'), ... % Available file formats
        'AnatIn'};                         % DefaultFormats: {ChannelIn,DataIn,DipolesIn,EventsIn,AnatIn,MriIn,NoiseCovIn,ResultsIn,SspIn,SurfaceIn,TimefreqIn}
    % Multichannel
    sProcess.options.multichannel.Comment = 'Multichannel';
    sProcess.options.multichannel.Type    = 'checkbox';
    sProcess.options.multichannel.Value   = 0;
    % Multichannel
    sProcess.options.softgt.Comment = 'Soft groundtruth';
    sProcess.options.softgt.Type    = 'checkbox';
    sProcess.options.softgt.Value   = 0;
    % Method: Average or PCA
    sProcess.options.label2.Comment = '<BR>Slice Axis:';
    sProcess.options.label2.Type    = 'label';
    sProcess.options.sliceaxis.Comment = {'Axial'; 'Sagittal'; 'Coronal'};
    sProcess.options.sliceaxis.Type    = 'radio';
    sProcess.options.sliceaxis.Value   = 1;
    
    % Event name
    sProcess.options.loss.Comment = 'Loss function: ';
    sProcess.options.loss.Type    = 'text';
    sProcess.options.loss.Value   = 'DiceLoss';
    
    % Multichannel
    sProcess.options.label4.Comment = '<BR>Uncertainty';
    sProcess.options.label4.Type    = 'label';
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
    
    %% Get all subjects to create the participants.tsv file
    subjects = {};
    for i = 1:length(sInputs)
        subjects{i} = str_remove_spec_chars(sInputs(i).SubjectName);
    end
    
    export_participants_tsv(unique(subjects))
    
    
    
    % === OUTPUT STUDY ===
    
    % Prepare parallel pool, if requested
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
    
    %% Convert the input trials to NIFTI files
    filenames = cell(length(sInputs),1);
    for iFile = 1:length(sInputs)
        sInput = sInputs(iFile);
        
        filenames{iFile} = convertTopography2matrix(sInput, sProcess, wanted_Fs, iFile);
        
        
    end
    OutputFiles = {};
end


function OutputMriFile = convertTopography2matrix(sInput, sProcess, wanted_Fs, iEpoch)
%   % Ignoring the bad sensors in the interpolation, so some values will be interpolated from the good sensors
%   WExtrap = GetInterpolation(iDS, iFig, TopoInfo, Vertices, Faces, bfs_center, bfs_radius, chan_loc(selChan,:));
% 
%         
%   [DataToPlot, Time, selChan, overlayLabels, dispNames, StatThreshUnder, StatThreshOver] = GetFigureData(iDS, iFig, 0);

    % TODO
    disp('DOES The colormap need to be GRAY???')
    
    dataMat = in_bst(sInput.FileName);
    current_Fs = round(1/diff(dataMat.Time(1:2)));
    
    %% ADD A JITTER
    % We want to avoid the model learning the positioning of the event so
    % we crop the time dimension on both sides with a jitter
    
    discardElementsBeginning = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);
    discardElementsEnd = round(randi(sProcess.options.jitter.Value{1}*1000)/1000 * current_Fs);
    
    dataMat.Time = dataMat.Time(1+discardElementsBeginning:end-discardElementsEnd);
    dataMat.F = dataMat.F(:,1+discardElementsBeginning:end-discardElementsEnd);
    
    %% Resample if needed
    if ~isnan(wanted_Fs) && current_Fs~=wanted_Fs
        %[x, time_out] = process_resample('Compute', x, time_in, NewRate)
        [dataMat.F, dataMat.Time] = process_resample('Compute', dataMat.F, dataMat.Time, wanted_Fs);
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
    

    % Get output study
    [tmp, iStudy] = bst_process('GetOutputStudy', sProcess, sInput);
    tfOPTIONS.iTargetStudy = iStudy;

    % Get channel file
    sChannel = bst_get('ChannelForStudy', iStudy);
    % Load channel file
    ChannelMat = in_bst_channel(sChannel.FileName);

    % Select the appropriate sensors
    nElectrodes = 0;
    selectedChannels = [];
    for iChannel = 1:length(ChannelMat.Channel)
       if strcmp(ChannelMat.Channel(iChannel).Type, 'EEG') || strcmp(ChannelMat.Channel(iChannel).Type, 'MEG')  %% ACCOMMODATE MORE HERE - fNIRS?
          nElectrodes = nElectrodes + 1;
          selectedChannels(end + 1) = iChannel;
       end
    end
    
    %% Gather the topography slices to a single 3d matrix
    % Here the time dimension is the 3rd dimension
    open_close_topography_window(sInput, 'open')
    NIFTI = channelMatrix2pixelMatrix(dataMat.F, dataMat.Time, selectedChannels);
    open_close_topography_window(sInput, 'close')

    %% Get the output filename and 
    BstMriFile = '/home/nas/Consulting/brainstorm_db/ivado@brainstorm/anat/@default_subject/subjectimage_T1.mat';
    
    % Use default anatomy (IS THIS SOMETHING IMPORTANT TO CONSIDER CHANGING - MAYBE FOR SOURCE LOCALIZATION ON MEG STUDIES???)
    % TODO - CONSIDER ADDING THE INDIVIDUAL ANATOMY HERE
    sMri = load(bst_fullfile(bst_get('BrainstormHomeDir'), 'defaults', 'anatomy', 'ICBM152', 'subjectimage_T1.mat'));
    
    % Substitute the voxels with the 2D slices created from the 2dlayout
    % topography
    sMri.Cube = NIFTI;
    
    % Output
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                           'IvadomedNiftiFiles', ...
                           protocol.Comment);
       
    % No special characters to avoid messing up with the IVADOMED importer
    subject = str_remove_spec_chars(sInput.SubjectName);
    session = str_remove_spec_chars(sInput.Condition);
                       
    % Hack to accommodate ivadomed derivative selection:
    % https://github.com/ivadomed/ivadomed/blob/master/ivadomed/loader/utils.py # L812
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'; % This hack accommodates up to 240 trials - for more find another solution 
                                            % - like double letters (not the same though or the same IVADOMED loader problem would occur)
    iLetter = floor(iEpoch/10);
    
    if iLetter == 0
        iEpoch = num2str(iEpoch);
    else
        iEpoch = [letters(iLetter) num2str(iEpoch)];
    end
    
    % TODO - IF MULTIPLE SUBJECTS OR MULTIPLE SESSIONS - ACCOMMODATE THE MAIN
    % FOLDER STRUCTURE
    OutputMriFile = bst_fullfile(parentPath, ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_epoch' iEpoch '.nii']);

    %% Export the created cube to NIFTI
    OutputMriFile = export2NIFTI(sMri, OutputMriFile);
    
    %% Create derivative
    F_derivative = ones(size(dataMat.F));  % I start with ones instead of zeros. The annotations will be marked as 0.
                                           % The reason for this is that
                                           % the saved images are inverted

    % The derivative will be based on a period of time that is annotated to
    % be the Ground truth.
    % In the case of extended event, only that period of time will annotated
    % In the case of simpple event, the selection 
    iAllSelectedEvents = find(ismember({dataMat.Events.label}, strsplit(sProcess.options.eventname.Value,{',',' '})));
    annotationValue = 0;
    for iSelectedEvent = iAllSelectedEvents
        annotationValue = annotationValue-1;
        % EXTENDED EVENTS
        for iEvent = 1:size(dataMat.Events(iSelectedEvent).times,2)
            iAnnotation_time_edges  = bst_closest(dataMat.Events(iSelectedEvent).times(:,iEvent)', dataMat.Time);

            % If no specific channels are annotated, annotate the entire slice
            if isempty(dataMat.Events(iSelectedEvent).channels{1,iEvent})
                F_derivative(:,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
            else
                iAnnotation_channels  = find(ismember({ChannelMat.Channel.Name}, dataMat.Events(iSelectedEvent).channels{1,iEvent}));
                F_derivative(iAnnotation_channels,iAnnotation_time_edges(1):iAnnotation_time_edges(2)) = annotationValue;
            end
        end
    end
        
    open_close_topography_window(sInput, 'open')
    NIFTI_derivative = channelMatrix2pixelMatrix(F_derivative, dataMat.Time, selectedChannels);
    open_close_topography_window(sInput, 'close')

    
    % Set the values to 0 and 1 for the annotations
    % Hard threshold
    NIFTI_derivative = NIFTI_derivative/max(max(max(NIFTI_derivative)));
    NIFTI_derivative(NIFTI_derivative<0.5) = 0;
    NIFTI_derivative(NIFTI_derivative>=0.5) = 1;
    
    % Annotate derivative
    sMri.Cube = NIFTI_derivative;
    
    % TODO - IF MULTIPLE SUBJECTS OR MULTIPLE SESSIONS - ACCOMMODATE THE MAIN
    % FOLDER STRUCTURE
    OutputDerivativeMriFile = bst_fullfile(parentPath, 'derivatives', 'labels', ['sub-' subject], ['ses-' session], 'anat', ['sub-' subject '_ses-' session '_epoch' iEpoch '_' sProcess.options.eventname.Value '.nii']);
    
    %% Export the created cube to NIFTI
    OutputMriFile = export2NIFTI(sMri, OutputDerivativeMriFile);
    
end

function open_close_topography_window(sInput, action)
    global GlobalData
    if strcmp(action, 'open')
        %% Open a window to inherit properties
        %[hFig, iDS, iFig] = view_topography(DataFile, Modality, TopoType, F)
        % TODO - consider adding flag on view_topography for not displaying the
        % figure when it is for Ivadomed
        % Modality       : {'MEG', 'MEG GRAD', 'MEG MAG', 'EEG', 'ECOG', 'SEEG', 'NIRS'}
        % TopoType       : {'3DSensorCap', '2DDisc', '2DSensorCap', 2DLayout', '3DElectrodes', '3DElectrodes-Cortex', '3DElectrodes-Head', '3DElectrodes-MRI', '3DOptodes', '2DElectrodes'}

            % TODO - GET MODALITY AUTOMATICALLY
        [hFig, iDS, iFig] = view_topography(sInput.FileName, 'MEG', '2DSensorCap');        
        [hFig, iFig, iDS] = bst_figures('GetFigure', GlobalData.DataSet.Figure.hFigure);
        set(hFig, 'Visible', 'off');
    elseif strcmp(action, 'close')
        % Close window
        % This needs to be done since the resizing of multiple windows that 
        % Brainstorm has affects the NIFTI files produced.
        % Only one window can be open at a time.
        % This affects parallel processing of this function.
        % TODO - Consider a workaround
        close(GlobalData.DataSet.Figure.hFigure)
    end
end


function NIFTI = channelMatrix2pixelMatrix(F, Time, selectedChannels)
    global GlobalData

%      %  %  %  MAKE SURE TO MAKE ALL THE VALUES POSITIVE AND CHANGE THE
%         %  MIN MAX TO [0, MIN+MAX]
% THIS IS STILL NOT WORKING
%     F = abs(min(min(F(selectedChannels,:)))) + F;
    
    
    % GLOBAL MIN_MAX FOR EACH TRIAL
    the_min = min(min(F(selectedChannels,:)));
    the_max = max(max(F(selectedChannels,:)));
    
    % This is altering the EEG 2D display - NOT THE COLORBAR ON THE BOTTOM
    % RIGHT - THE COLORBAR NEEDS TO BE ADDRESSED
    GlobalData.Dataset.Figure.Handles.DataMinMax = [the_min, the_max];
    
    
    % Get size of exported files
    [height,width,~] = size(print(GlobalData.DataSet.Figure.hFigure, '-noui', '-r50', '-RGBImage'));

    
    NIFTI = zeros(height, width, length(Time), 'uint8');
    for iTime = 1:length(Time)
        DataToPlot = F(selectedChannels,iTime);

        % ===== APPLY TRANSFORMATION =====
        % Mapping on a different surface (magnetic source reconstruction of just smooth display)
        if ~isempty(GlobalData.DataSet.Figure.Handles.Wmat)
            % Apply interpolation matrix sensors => display surface
            if (size(GlobalData.DataSet.Figure.Handles.Wmat,1) == length(DataToPlot))
                DataToPlot = full(GlobalData.DataSet.Figure.Handles.Wmat * DataToPlot);
            % Find first corresponding indices
            else
%                 [tmp,I,J] = intersect(selectedChannels, GlobalData.DataSet(iDS).Figure(iFig).SelectedChannels);
                [tmp,I,J] = intersect(selectedChannels, GlobalData.DataSet.Figure.SelectedChannels);
                DataToPlot = full(GlobalData.DataSet.Figure.Handles.Wmat(:,J) * DataToPlot(I));
            end
        end         

        set(GlobalData.DataSet.Figure.Handles.hSurf, 'FaceVertexCData', DataToPlot, 'EdgeColor', 'none');

        % Check exporting image
        img = print(GlobalData.DataSet.Figure.hFigure, '-noui', '-r50', '-RGBImage');        
        img_gray= 255-rgb2gray(img); % Inverse black/white to have the surrounding black
        NIFTI(:,:,iTime) = img_gray;
    end
    
    
    %% CROP IMAGE (AVOID COLORBAR AND WASTED SPACE AROUND TOPOGRAPHY)
    % ALSO FLIP TO CREATE CORRECT ORIENTATION WITH
    % LEFT-RIGHT-ANTERIOR-POSTERIOR
    crop_from_top = 30;
    crop_from_bottom = -30;
    crop_from_left = 30;
    crop_from_right = -60;
    
    NIFTI = NIFTI(crop_from_top:end+crop_from_bottom, crop_from_left:end+crop_from_right,:);
    
    NIFTI = flip(permute(NIFTI,[2,1,3]),2);
%     figure(1);
%     imagesc(squeeze(NIFTI2(:,:,1)));
    
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


function export_participants_tsv(subjects)
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                           'IvadomedNiftiFiles', ...
                           protocol.Comment);
                       
    if ~exist(parentPath, 'dir')  % This avoids a warning if the folder already exists
        mkdir(parentPath)
    end    
    
    participants_data = struct;
    
    for i = 1:length(subjects)
        participants_data(i).participant_id = subjects{i};
        participants_data(i).sex = 'na';
        participants_data(i).age = 'na';
    end
        
    % Writetable didn't allow export in .tsv - I rename it after
    writetable(struct2table(participants_data), bst_fullfile(parentPath, 'participants.txt'), 'Delimiter', ',')
    movefile(bst_fullfile(parentPath, 'participants.txt'), bst_fullfile(parentPath, 'participants.tsv'))
end


function export_participants_json()
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                           'IvadomedNiftiFiles', ...
                           protocol.Comment);


    text = '{\n"participant_id": {\n\t"Description": "Unique ID",\n\t"LongName": "Participant ID"\n\t},\n"sex": {\n\t"Description": "M or F",\n\t"LongName": "Participant sex"\n\t},\n"age": {\n\t"Description": "yy",\n\t"LongName": "Participant age"\n\t}\n}';

    fileID = fopen(bst_fullfile(parentPath, 'participants.json'),'w');
    fprintf(fileID,text);
    fclose(fileID);


end


function export_dataset_description()
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                           'IvadomedNiftiFiles', ...
                           protocol.Comment);

    text = '{\n\t"BIDSVersion": "1.6.0",\n\t"Name": "Ivadomed@Brainstorm"\n}';
    
    fileID = fopen(bst_fullfile(parentPath, 'dataset_description.json'),'w');
    fprintf(fileID,text);
    fclose(fileID);
end


function export_readme()
    protocol = bst_get('ProtocolInfo');
    parentPath = bst_fullfile(bst_get('BrainstormTmpDir'), ...
                           'IvadomedNiftiFiles', ...
                           protocol.Comment);

    text = 'Converted BIDS dataset from Brainstorm trials';
    
    fileID = fopen(bst_fullfile(parentPath, 'README'),'w');
    fprintf(fileID,text);
    fclose(fileID);
end




