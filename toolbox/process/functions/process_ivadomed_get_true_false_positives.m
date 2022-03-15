function varargout = process_ivadomed_get_true_false_positives( varargin )
% PROCESS_GET_TRUE_FALSE_POSITIVES: this function compares the outputs of a
% model to the ground truth events
% USAGE:    sProcess = process_ivadomed_create_dataset('GetDescription')
%        OutputFiles = process_ivadomed_create_dataset('Run', sProcess, sInput)

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
% Author: Konstantinos Nasiotis, 2022

eval(macro_method);
end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<*DEFNU>
    % Description the process
    sProcess.Comment     = 'Compare ground truth - model predictions';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3118;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'raw', 'data'};
    sProcess.OutputTypes = {'raw', 'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    
    % Event name
    sProcess.options.gt_eventname.Comment = 'Ground truth event';
    sProcess.options.gt_eventname.Type    = 'text';
    sProcess.options.gt_eventname.Value   = 'event1';
    
    % Event name
    sProcess.options.prediction_eventname.Comment = 'Model annotated event';
    sProcess.options.prediction_eventname.Type    = 'text';
    sProcess.options.prediction_eventname.Value   = 'event2';
    
    % Options: Segment around spike
    sProcess.options.timewindow_annot.Comment  = 'Leniency period: ';
    sProcess.options.timewindow_annot.Type     = 'value';
    sProcess.options.timewindow_annot.Value    = {0.020,'ms',0};
    
    % Create false_positive_negative events? (Only for raw files)
    sProcess.options.false_pos_neg.Comment = 'Annotate false positives/negatives';
    sProcess.options.false_pos_neg.Type    = 'checkbox';
    sProcess.options.false_pos_neg.Value   = 0;
    
    % Event help comment
    sProcess.options.false_pos_neg_help.Comment = '<I><FONT color="#777777"> Creates new events (Only for links to raw files)</FONT></I>';
    sProcess.options.false_pos_neg_help.Type    = 'label';
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess)
    Comment = 'Ivadomed - Compare prediction to ground truth';
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs)
    
    % Check if dealing with trials or links to raw files
    [sStudy, iStudy, iData] = bst_get('DataFile', sInputs(1).FileName);
    isRaw = strcmpi(sStudy.Data(iData).DataType, 'raw');

     for iInput = 1:length(sInputs) % Link to raw files inputs

        % Read file descriptor
        DataMat = in_bst_data(sInputs(iInput).FileName);

        if isRaw
            sFile = DataMat.F;
        else
            sFile = in_fopen(sInputs(iInput).FileName, 'BST-DATA');
        end
        events = sFile.events;        

        % Check if both selected events exist
         disp(1)
         if ~all(ismember({sProcess.options.gt_eventname.Value, sProcess.options.prediction_eventname.Value}, {events.label}))
            warning('The selected events are not present in this file. Skipping')
         else
             % Create confusion matrix
             events = confusion_matrix(events, sProcess.options.gt_eventname.Value, sProcess.options.prediction_eventname.Value, sProcess.options.timewindow_annot.Value{1});
             
             % If selected, update the link to raw file events with false
             % positive and negative events
             if sProcess.options.false_pos_neg.Value && isRaw
                
                DataMat.F.events = events;
        
                ProtocolInfo = bst_get('ProtocolInfo');
                bst_save(bst_fullfile(ProtocolInfo.STUDIES, sInputs(iInput).FileName), DataMat, 'v6');
        
             end
         end
            
     end
     OutputFiles = [];
end



function events = confusion_matrix(events, gt_label, model_annot_label, leniency)

    gt_times = events(find(ismember({events.label}, gt_label ))).times;
    model_times = events(find(ismember({events.label}, model_annot_label ))).times;
    
    % Convert 
    if size(model_times,1) == 2  %(extended events)
        model_times = mean(model_times,1);
    end
    
    gt = zeros(1, length(gt_times));
    for iEvent = 1:length(gt_times)
        iClosest = bst_closest(gt_times(iEvent), model_times);
        
        if abs(model_times(iClosest) - gt_times(iEvent))<leniency
            gt(iEvent) = 1;
        end
    end
    
    pred = zeros(1, length(model_times));
    for iEvent = 1:length(model_times)
        iClosest = bst_closest(model_times(iEvent), gt_times);
        
        if abs(gt_times(iClosest) - model_times(iEvent)) < leniency
            pred(iEvent) = 1;
        end
    end
    
    temp_events = struct;
    
    temp_events(1).label = 'true_positives';
    temp_events(2).label = 'false_positives';
    temp_events(3).label = 'false_negatives';
    
    temp_events(1).times = gt_times(find(gt));
    temp_events(2).times = model_times(find(~pred));
    temp_events(3).times = gt_times(find(~gt));
    
    temp_events(1).color = [0 1 0];
    temp_events(2).color = [1 0 0];
    temp_events(3).color = [1 0 0];
    
    ii = 1;
    for iEvent = 1:3
        if ~isempty(temp_events(iEvent).times)
            newEvents(ii).label      = temp_events(iEvent).label
            newEvents(ii).color      = temp_events(iEvent).color;
            newEvents(ii).times      = temp_events(iEvent).times;
            newEvents(ii).reactTimes = [];
            newEvents(ii).select     = 1;
            newEvents(ii).epochs     = ones(1, size(newEvents(ii).times, 2));
            newEvents(ii).channels   = cell(1, size(newEvents(ii).times, 2));
            newEvents(ii).notes      = cell(1, size(newEvents(ii).times, 2));
            ii = ii+1;
        end
    end
    
    events = [events newEvents];
end
