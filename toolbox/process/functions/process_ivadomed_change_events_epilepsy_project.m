function varargout = process_ivadomed_change_events_epilepsy_project( varargin )
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
    sProcess.Comment     = 'Change epilepsy events to prepare for Ivadomed';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'IvadoMed Toolbox';
    sProcess.Index       = 3119;
    sProcess.Description = 'https://ivadomed.org/en/latest/index.html';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'raw', 'data'};
    sProcess.OutputTypes = {'raw', 'data'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    % Recordings time window
    sProcess.options.triallength.Comment = 'Trial length:';
    sProcess.options.triallength.Type    = 'value';
    sProcess.options.triallength.Value   = {5, 'sec', 0};
    % Sliding window 
    sProcess.options.slidingTrialOverlap.Comment = 'Sliding trials overlap';
    sProcess.options.slidingTrialOverlap.Type    = 'value';
    sProcess.options.slidingTrialOverlap.Value   = {0, '%', 0};
    
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess)
    Comment = 'Change events for epilepsy project';
end


%% ===== RUN =====
function OutputFiles = Run(sProcess, sInputs)

    for iInput = 1:length(sInputs) % Link to raw files inputs

        
        % Read file descriptor
        DataMat = in_bst_data(sInputs(iInput).FileName);
 
        sFile = DataMat.F;
        
        %% Get channelMat
          % Get output study
        [tmp, iStudy] = bst_process('GetOutputStudy', sProcess, sInputs(iInput));
        % Get channel file
        sChannel = bst_get('ChannelForStudy', iStudy);
        % Load channel file
        ChannelMat = in_bst_channel(sChannel.FileName);
        
        %% Get original events
        events = sFile.events;        
        
        %% New events every 5 seconds - this will be used to create new trials
        new5SecondsEvents = struct;
        
        seconds_5 = 0:sProcess.options.triallength.Value{1}*(100-sProcess.options.slidingTrialOverlap.Value{1})/100:2000;  % It doesn't matter that the events are longer than the acquisition system
        event_name = sprintf('spaced_out_%d', sProcess.options.triallength.Value{1});
        ii = 1;
        new5SecondsEvents(ii).label      = event_name;
        new5SecondsEvents(ii).color      = rand(1,3);
        new5SecondsEvents(ii).times      = seconds_5;
        new5SecondsEvents(ii).reactTimes = [];
        new5SecondsEvents(ii).select     = 1;
        new5SecondsEvents(ii).epochs     = ones(1, size(new5SecondsEvents(ii).times, 2));
        new5SecondsEvents(ii).channels   = cell(1, size(new5SecondsEvents(ii).times, 2));
        new5SecondsEvents(ii).notes      = cell(1, size(new5SecondsEvents(ii).times, 2));
        ii = ii+1;
        
        
        %% Remove all previous entries of the IEDs keywords
        ied_keywords = {'spikeandwave', 'sharp', 'spikesharpandwave', 'spike', 'polyspikeandwave', 'polyspike'};

        iEventsToRemove = find(ismember({events.label}, ied_keywords));
        events(iEventsToRemove) = [];
        
        %% Find which events correspond to what Ellie annotated, and assign the channel to them
        % The annotations follow this template: channelLabels_iedLabel, e.g.
        % FP1_spikeandwave, or O1_O2_polyspike
        all_labels = {events.label};
        
        ied_newEvents = repmat(db_template('event'), 0);

        iLabelQualifies = false(length(all_labels), 1);
        for iLabel = 1:length(all_labels)
            
            stripped = strsplit(all_labels{iLabel},'_');
            each_element_check = false(length(stripped),1);
            
            if length(stripped) < 2
                continue
            else
                for i = 1:length(stripped)-1
                    each_element_check(i) = any(ismember(lower({ChannelMat.Channel.Name}), lower(stripped{i})));
                end
                each_element_check(end) = any(ismember(ied_keywords, stripped{end}));
            end
            if all(each_element_check)
                iLabelQualifies(iLabel) = true;
                events(iLabel).channels = repmat({{stripped{1,1:end-1}}},1, length(events(iLabel).times));
                
                iIED = find(ismember(ied_keywords, stripped{end}));
                
                ied_newEvents(iIED).label = ied_keywords{iIED};
                ied_newEvents(iIED).color = rand(1,3);
                ied_newEvents(iIED).reactTimes = [];
                ied_newEvents(iIED).select = 1;
                
                fields = {'epochs', 'times', 'channels', 'notes'};
                for iField = 1:length(fields)
                    ied_newEvents(iIED).(fields{iField}) = [ied_newEvents(iIED).(fields{iField}) events(iLabel).(fields{iField})];
                end
                
            end
        end
        
        % Remove empty entries
        iEntries_to_keep = ~cellfun(@isempty, {ied_newEvents.label});
        ied_newEvents = ied_newEvents(iEntries_to_keep);
        
        % Concatenate with the spaced_out_5
        if ~ismember(event_name, {events.label})
            events = [events new5SecondsEvents ied_newEvents];
        else
            events = [events ied_newEvents];
        end

        
        %% Some converted files accidentally concatenated the spaced_out_5
        % events twice. Correct this here
        iEvents = find(ismember({events.label}, event_name));
        
        if length(iEvents)>1
            events(iEvents(2:end)) = [];
        end
            
        %% Import this list
        DataMat.F.events = events;
        
        ProtocolInfo = bst_get('ProtocolInfo');
        bst_save(bst_fullfile(ProtocolInfo.STUDIES, sInputs(iInput).FileName), DataMat, 'v6');

    end
    OutputFiles = [];
end


     
     
     
     
     
     
     
     
     
     