function ivadomed_deactivate_bids_validator(fname)
    % Ivadomed uses a BIDS validator for making sure that the dataset that
    % is used as an input is proper
    % However the datasets that are created from the Brainstorm functions
    % have filenames that are not BIDS compatible.
    % This scripts deactivates the validator by changing the python
    % function that contains the call to the validator
    
    % Read and modify
    fid = fopen(fname,'rt');
    idx = 1;
    optionCell = {};
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'indexer = pybids.BIDSLayoutIndexer(force_index=force_index)')
            line = strrep(line, 'indexer = pybids.BIDSLayoutIndexer(force_index=force_index)', 'indexer = pybids.BIDSLayoutIndexer(force_index=force_index, validate=False)');
        end
        optionCell{idx,1} = line;
        idx = idx + 1;
    end
    fclose(fid);

    % Write to file
    fid = fopen(fname,'w');
    for iLine = 1:length(optionCell)
        fprintf(fid, '%s\n', optionCell{iLine});
    end
    fclose(fid);
    
    disp('BIDS validator has been deactivated within the Plugin')
end