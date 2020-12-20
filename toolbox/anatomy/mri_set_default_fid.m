function sMri = mri_set_default_fid(sMri, Method)
% MRI_SET_DEFAULT_FID:  Set default fiducials based on the MNI transformation.
%
% INPUTS:
%    - sMri   : Brainstorm MRI structure
%    - Method : Method used for the MNI normalization
%               'maff8': Affine transform obtained with bst_normalize_mni.m
%               'cat12': Non-linear y_ and iy_ deformation fields generated by CAT12 

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
% Authors: Francois Tadel, 2017-2020

% Parse inputs
if (nargin < 2) || isempty(Method)
    Method = 'maff8';
end

% ===== NCS FIDUCIALS =====
% MNI coordinates for all the fiducials
switch (Method)
    case 'maff8'
        AC  = [0,   3,  -4] ./ 1000;
        PC  = [0, -25,  -2] ./ 1000;
        IH  = [0, -10,  60] ./ 1000;
        Orig= [0,   0,   0];    
        NAS = [ 0,   84, -50] ./ 1000;
        LPA = [-83, -19, -48] ./ 1000;
        RPA = [ 83, -19, -48] ./ 1000;
    case 'cat12'    % Obtained to get exactly the same points as previously (maff8)
        AC  = [  0.73,   2.73,  -2.76] ./ 1000;
        PC  = [  0.59, -25.52,  -0.15] ./ 1000;
        IH  = [  0.98, -10.76,  60.96] ./ 1000;
        Orig= [  0.73,  -0.40,   1.12] ./ 1000;    
        NAS = [  1.86,  73.74, -40.51] ./ 1000;
        LPA = [-80.04, -24.16, -44.85] ./ 1000;
        RPA = [ 81.87, -26.83, -44.35] ./ 1000;
end
% Convert: MNI (meters) => MRI (millimeters)
sMri.NCS.AC     = cs_convert(sMri, 'mni', 'mri', AC) .* 1000;
sMri.NCS.PC     = cs_convert(sMri, 'mni', 'mri', PC) .* 1000;
sMri.NCS.IH     = cs_convert(sMri, 'mni', 'mri', IH) .* 1000;
sMri.NCS.Origin = cs_convert(sMri, 'mni', 'mri', Orig) .* 1000;

% ===== SCS FIDUCIALS =====
% Compute default positions for NAS/LPA/RPA if not available yet
if ~isfield(sMri, 'SCS') || ~isfield(sMri.SCS, 'NAS') || ~isfield(sMri.SCS, 'LPA') || ~isfield(sMri.SCS, 'RPA') ...
        || isempty(sMri.SCS.NAS) || isempty(sMri.SCS.LPA) || isempty(sMri.SCS.RPA) 
    sMri.SCS.NAS = cs_convert(sMri, 'mni', 'mri', NAS) .* 1000;
    sMri.SCS.LPA = cs_convert(sMri, 'mni', 'mri', LPA) .* 1000;
    sMri.SCS.RPA = cs_convert(sMri, 'mni', 'mri', RPA) .* 1000;
    % Compute SCS transformation, if not available
    if ~isfield(sMri.SCS, 'R') || ~isfield(sMri.SCS, 'T') || isempty(sMri.SCS.R) || isempty(sMri.SCS.T)
        [Transf, sMri] = cs_compute(sMri, 'SCS');
    end
end


