function clusters = getNchannelsAroundCenter(nChannels, ChannelMat, FigureObject)
    % 
    x = FigureObject.Handles.MarkersLocs(:,1);
    y = FigureObject.Handles.MarkersLocs(:,2);
    
    
    clusters = struct;
    
    for iCluster = 1:2
        
        center_x = min(x) + (abs(min(x)) + abs(max(x)))*rand(1,1);
        center_y = min(y) + (abs(min(y)) + abs(max(y)))*rand(1,1);

        clusters(iCluster).CenterCoords = [center_x center_y];
        
        
        distance = zeros(length(x),1);
        for i = 1:length(x)
            distance(i) = sqrt((center_x-x(i))^2 + (center_y-y(i))^2);
        end

        [a,sortedChannelOrder] = sort(distance);

        indices = sortedChannelOrder(1:nChannels);

        xx = x(indices);
        yy = y(indices);
        
        clusters(iCluster).SelectedChannelsNames = {ChannelMat.Channel(FigureObject.SelectedChannels(indices)).Name};
        clusters(iCluster).SelectedChannelsCoords = [xx yy];
        clusters(iCluster).SelectedIndicesOnChannelMat = FigureObject.SelectedChannels(indices);
    end
end

