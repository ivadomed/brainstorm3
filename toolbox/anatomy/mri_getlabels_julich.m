function Labels = mri_getlabels_julich()
% ATLAS     : Juelich histological atlas (Eickhoff 2005)
% REFERENCE : http://www.fz-juelich.de/inm/inm-1/EN/Forschung/JuBrain/Jubrain_Webtools/Jubrain_Webtools_node.html

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

Labels = {...
        0, 'Background',[    0    0    0]; ...
        1, 'TE1.1 L',   [    0  204    0]; ...   % Left Area TE 1.1 (HESCHL)
        2, 'TE1.1 R',   [    0  204    0]; ...   % Right Area TE 1.1 (HESCHL)
        3, '3b L',      [  102  102  255]; ...   % Left Area 3b (PostCG)
        4, '3b R',      [  102  102  255]; ...   % Right Area 3b (PostCG)
        5, 'hOc4la L',  [    0  255  255]; ...   % Left Area hOc4la (LOC)
        6, 'hOc4la R',  [    0  255  255]; ...   % Right Area hOc4la (LOC)
        7, 'PF L',      [  102    0    0]; ...   % Left Area PF (IPL)
        8, 'PF R',      [  102    0    0]; ...   % Right Area PF (IPL)
        9, 'hOc2 L',    [  255  215    0]; ...   % Left Area hOc2 (V2, 18)
       10, 'hOc2 R',    [  255  215    0]; ...   % Right Area hOc2 (V2, 18)
       11, '6d2 L',     [  255    0    0]; ...   % Left Area 6d2 (PreCG)
       12, '6d2 R',     [  255    0    0]; ...   % Right Area 6d2 (PreCG)
       13, 'Ig3 L',     [  255  177  100]; ...   % Left Area Ig3 (Insula)
       14, 'Ig3 R',     [  255  177  100]; ...   % Right Area Ig3 (Insula)
       15, 'hOc4d L',   [  255    0  255]; ...   % Left Area hOc4d (Cuneus)
       16, 'hOc4d R',   [  255    0  255]; ...   % Right Area hOc4d (Cuneus)
       17, 'Fo4 L',     [    0  102    0]; ...   % Left Area Fo4 (OFC)
       18, 'Fo4 R',     [    0  102    0]; ...   % Right Area Fo4 (OFC)
       19, 'OP4 L',     [    0  204    0]; ...   % Left Area OP4 (POperc)
       20, 'OP4 R',     [    0  204    0]; ...   % Right Area OP4 (POperc)
       21, '1 L',       [  102  102  255]; ...   % Left Area 1 (PostCG)
       22, '1 R',       [  102  102  255]; ...   % Right Area 1 (PostCG)
       23, 'Ch4 L',     [    0  255  255]; ...   % Left Ch 4 (Basal Forebrain)
       24, 'Ch4 R',     [    0  255  255]; ...   % Right Ch 4 (Basal Forebrain)
       25, 'GapF-I L',  [  102    0    0]; ...   % Left GapMap Frontal-I (GapMap)
       26, 'GapF-I R',  [  102    0    0]; ...   % Right GapMap Frontal-I (GapMap)
       27, 'hPO1 L',    [  255  215    0]; ...   % Left Area hPO1 (POS)
       28, 'hPO1 R',    [  255  215    0]; ...   % Right Area hPO1 (POS)
       29, 'DDN L',     [  255    0    0]; ...   % Left Dorsal Dentate Nucleus (Cerebellum)
       30, 'DDN R',     [  255    0    0]; ...   % Right Dorsal Dentate Nucleus (Cerebellum)
       31, 'OP2 L',     [  255  177  100]; ...   % Left Area OP2 (POperc)
       32, 'OP2 R',     [  255  177  100]; ...   % Right Area OP2 (POperc)
       33, 'TE1.0 L',   [  255    0  255]; ...   % Left Area TE 1.0 (HESCHL)
       34, 'TE1.0 R',   [  255    0  255]; ...   % Right Area TE 1.0 (HESCHL)
       35, 'STS2 L',    [    0  102    0]; ...   % Left Area STS2 (STS)
       36, 'STS2 R',    [    0  102    0]; ...   % Right Area STS2 (STS)
       37, 'Id3 L',     [    0  204    0]; ...   % Left Area Id3 (Insula)
       38, 'Id3 R',     [    0  204    0]; ...   % Right Area Id3 (Insula)
       39, '6mp L',     [  102  102  255]; ...   % Left Area 6mp (SMA, mesial SFG)
       40, '6mp R',     [  102  102  255]; ...   % Right Area 6mp (SMA, mesial SFG)
       41, 'DG L',      [    0  255  255]; ...   % Left DG (Hippocampus)
       42, 'DG R',      [    0  255  255]; ...   % Right DG (Hippocampus)
       43, 'PGp L',     [  102    0    0]; ...   % Left Area PGp (IPL)
       44, 'PGp R',     [  102    0    0]; ...   % Right Area PGp (IPL)
       45, 'Ig1 L',     [  255  215    0]; ...   % Left Area Ig1 (Insula)
       46, 'Ig1 R',     [  255  215    0]; ...   % Right Area Ig1 (Insula)
       47, '6ma L',     [  255    0    0]; ...   % Left Area 6ma (preSMA, mesial SFG)
       48, '6ma R',     [  255    0    0]; ...   % Right Area 6ma (preSMA, mesial SFG)
       49, 'TI L',      [  255  177  100]; ...   % Left Area TI (STG)
       50, 'TI R',      [  255  177  100]; ...   % Right Area TI (STG)
       51, 'PFt L',     [  255    0  255]; ...   % Left Area PFt (IPL)
       52, 'PFt R',     [  255    0  255]; ...   % Right Area PFt (IPL)
       53, 'p24ab L',   [    0  102    0]; ...   % Left Area p24ab (pACC)
       54, 'p24ab R',   [    0  102    0]; ...   % Right Area p24ab (pACC)
       55, 'p24c L',    [    0  204    0]; ...   % Left Area p24c (pACC)
       56, 'p24c R',    [    0  204    0]; ...   % Right Area p24c (pACC)
       57, 'hOc3v L',   [  102  102  255]; ...   % Left Area hOc3v (LingG)
       58, 'hOc3v R',   [  102  102  255]; ...   % Right Area hOc3v (LingG)
       59, 'TE3 L',     [    0  255  255]; ...   % Left Area TE 3 (STG)
       60, 'TE3 R',     [    0  255  255]; ...   % Right Area TE 3 (STG)
       61, 'PFcm L',    [  102    0    0]; ...   % Left Area PFcm (IPL)
       62, 'PFcm R',    [  102    0    0]; ...   % Right Area PFcm (IPL)
       63, 'GapF-O L',  [  255  215    0]; ...   % Left GapMap Frontal-to-Occipital (GapMap)
       64, 'GapF-O R',  [  255  215    0]; ...   % Right GapMap Frontal-to-Occipital (GapMap)
       65, '7P L',      [  255    0    0]; ...   % Left Area 7P (SPL)
       66, '7P R',      [  255    0    0]; ...   % Right Area 7P (SPL)
       67, 'hIP5 L',    [  255  177  100]; ...   % Left Area hIP5 (IPS)
       68, 'hIP5 R',    [  255  177  100]; ...   % Right Area hIP5 (IPS)
       69, '6d1 L',     [  255    0  255]; ...   % Left Area 6d1 (PreCG)
       70, '6d1 R',     [  255    0  255]; ...   % Right Area 6d1 (PreCG)
       71, 'FG1 L',     [    0  102    0]; ...   % Left Area FG1 (FusG)
       72, 'FG1 R',     [    0  102    0]; ...   % Right Area FG1 (FusG)
       73, 'hIP4 L',    [    0  204    0]; ...   % Left Area hIP4 (IPS)
       74, 'hIP4 R',    [    0  204    0]; ...   % Right Area hIP4 (IPS)
       75, 'OP6 L',     [  102  102  255]; ...   % Left Area OP6 (Frontal Operculum)
       76, 'OP6 R',     [  102  102  255]; ...   % Right Area OP6 (Frontal Operculum)
       77, 'FG2 L',     [    0  255  255]; ...   % Left Area FG2 (FusG)
       78, 'FG2 R',     [    0  255  255]; ...   % Right Area FG2 (FusG)
       79, 'hIP6 L',    [  102    0    0]; ...   % Left Area hIP6 (IPS)
       80, 'hIP6 R',    [  102    0    0]; ...   % Right Area hIP6 (IPS)
       81, 'hIP3 L',    [  255  215    0]; ...   % Left Area hIP3 (IPS)
       82, 'hIP3 R',    [  255  215    0]; ...   % Right Area hIP3 (IPS)
       83, 'Fo5 L',     [  255    0    0]; ...   % Left Area Fo5 (OFC)
       84, 'Fo5 R',     [  255    0    0]; ...   % Right Area Fo5 (OFC)
       85, 'CM L',      [  255  177  100]; ...   % Left CM (Amygdala)
       86, 'CM R',      [  255  177  100]; ...   % Right CM (Amygdala)
       87, 'OP7 L',     [  255    0  255]; ...   % Left Area OP7 (Frontal Operculum)
       88, 'OP7 R',     [  255    0  255]; ...   % Right Area OP7 (Frontal Operculum)
       89, '4p L',      [    0  102    0]; ...   % Left Area 4p (PreCG)
       90, '4p R',      [    0  102    0]; ...   % Right Area 4p (PreCG)
       91, '33 L',      [    0  204    0]; ...   % Left Area 33 (ACC)
       92, '33 R',      [    0  204    0]; ...   % Right Area 33 (ACC)
       93, 'hOc6 L',    [  102  102  255]; ...   % Left Area hOc6 (POS)
       94, 'hOc6 R',    [  102  102  255]; ...   % Right Area hOc6 (POS)
       95, 'Ig2 L',     [    0  255  255]; ...   % Left Area Ig2 (Insula)
       96, 'Ig2 R',     [    0  255  255]; ...   % Right Area Ig2 (Insula)
       97, 'CA2 L',     [  102    0    0]; ...   % Left CA2 (Hippocampus)
       98, 'CA2 R',     [  102    0    0]; ...   % Right CA2 (Hippocampus)
       99, 'Fp2 L',     [  255  215    0]; ...   % Left Area Fp2 (FPole)
      100, 'Fp2 R',     [  255  215    0]; ...   % Right Area Fp2 (FPole)
      101, 'EC L',      [  255    0    0]; ...   % Left Entorhinal Cortex
      102, 'EC R',      [  255    0    0]; ...   % Right Entorhinal Cortex
      103, 'Id4 L',     [  255  177  100]; ...   % Left Area Id4 (Insula)
      104, 'Id4 R',     [  255  177  100]; ...   % Right Area Id4 (Insula)
      105, 'Id2 L',     [  255    0  255]; ...   % Left Area Id2 (Insula)
      106, 'Id2 R',     [  255    0  255]; ...   % Right Area Id2 (Insula)
      107, 'CA1 L',     [    0  102    0]; ...   % Left CA1 (Hippocampus)
      108, 'CA1 R',     [    0  102    0]; ...   % Right CA1 (Hippocampus)
      109, 'OP1 L',     [    0  204    0]; ...   % Left Area OP1 (POperc)
      110, 'OP1 R',     [    0  204    0]; ...   % Right Area OP1 (POperc)
      111, 'Id6 L',     [  102  102  255]; ...   % Left Area Id6 (Insula)
      112, 'Id6 R',     [  102  102  255]; ...   % Right Area Id6 (Insula)
      113, 'hIP8 L',    [    0  255  255]; ...   % Left Area hIP8 (IPS)
      114, 'hIP8 R',    [    0  255  255]; ...   % Right Area hIP8 (IPS)
      115, 'OP9 L',     [  102    0    0]; ...   % Left Area OP9 (Frontal Operculum)
      116, 'OP9 R',     [  102    0    0]; ...   % Right Area OP9 (Frontal Operculum)
      117, 'GapT-P L',  [  255  215    0]; ...   % Left GapMap Temporal-to-Parietal (GapMap)
      118, 'GapT-P R',  [  255  215    0]; ...   % Right GapMap Temporal-to-Parietal (GapMap)
      119, '6d3 L',     [  255    0    0]; ...   % Left Area 6d3 (SFS)
      120, '6d3 R',     [  255    0    0]; ...   % Right Area 6d3 (SFS)
      121, 'VDN L',     [  255  177  100]; ...   % Left Ventral Dentate Nucleus (Cerebellum)
      122, 'VDN R',     [  255  177  100]; ...   % Right Ventral Dentate Nucleus (Cerebellum)
      123, 'HATA L',    [  255    0  255]; ...   % Left HATA (Hippocampus)
      124, 'HATA R',    [  255    0  255]; ...   % Right HATA (Hippocampus)
      125, 'hIP2 L',    [    0  102    0]; ...   % Left Area hIP2 (IPS)
      126, 'hIP2 R',    [    0  102    0]; ...   % Right Area hIP2 (IPS)
      127, '44 L',      [    0  204    0]; ...   % Left Area 44 (IFG)
      128, '44 R',      [    0  204    0]; ...   % Right Area 44 (IFG)
      129, 'Fo7 L',     [  102  102  255]; ...   % Left Area Fo7 (OFC)
      130, 'Fo7 R',     [  102  102  255]; ...   % Right Area Fo7 (OFC)
      131, 'a29 L',     [    0  255  255]; ...   % Left Area a29 (retrosplenial)
      132, 'a29 R',     [    0  255  255]; ...   % Right Area a29 (retrosplenial)
      133, '45 L',      [  102    0    0]; ...   % Left Area 45 (IFG)
      134, '45 R',      [  102    0    0]; ...   % Right Area 45 (IFG)
      135, 'Id1 L',     [  255  215    0]; ...   % Left Area Id1 (Insula)
      136, 'Id1 R',     [  255  215    0]; ...   % Right Area Id1 (Insula)
      137, '5Ci L',     [  255    0    0]; ...   % Left Area 5Ci (SPL)
      138, '5Ci R',     [  255    0    0]; ...   % Right Area 5Ci (SPL)
      139, 'Id5 L',     [  255  177  100]; ...   % Left Area Id5 (Insula)
      140, 'Id5 R',     [  255  177  100]; ...   % Right Area Id5 (Insula)
      141, 'TE1.2 L',   [  255    0  255]; ...   % Left Area TE 1.2 (HESCHL)
      142, 'TE1.2 R',   [  255    0  255]; ...   % Right Area TE 1.2 (HESCHL)
      143, 'PGa L',     [    0  102    0]; ...   % Left Area PGa (IPL)
      144, 'PGa R',     [    0  102    0]; ...   % Right Area PGa (IPL)
      145, 'hIP1 L',    [    0  204    0]; ...   % Left Area hIP1 (IPS)
      146, 'hIP1 R',    [    0  204    0]; ...   % Right Area hIP1 (IPS)
      147, '2 L',       [  102  102  255]; ...   % Left Area 2 (PostCS)
      148, '2 R',       [  102  102  255]; ...   % Right Area 2 (PostCS)
      149, 'Ch123 L',   [    0  255  255]; ...   % Left Ch 123 (Basal Forebrain)
      150, 'Ch123 R',   [    0  255  255]; ...   % Right Ch 123 (Basal Forebrain)
      151, 'TE2.1 L',   [  102    0    0]; ...   % Left Area TE 2.1 (STG)
      152, 'TE2.1 R',   [  102    0    0]; ...   % Right Area TE 2.1 (STG)
      153, 'Ia L',      [  255  215    0]; ...   % Left Area Ia (Insula)
      154, 'Ia R',      [  255  215    0]; ...   % Right Area Ia (Insula)
      155, 'hOc3d L',   [  255    0    0]; ...   % Left Area hOc3d (Cuneus)
      156, 'hOc3d R',   [  255    0    0]; ...   % Right Area hOc3d (Cuneus)
      157, 'VTM L',     [  255  177  100]; ...   % Left VTM (Amygdala)
      158, 'VTM R',     [  255  177  100]; ...   % Right VTM (Amygdala)
      159, 'OP3 L',     [  255    0  255]; ...   % Left Area OP3 (POperc)
      160, 'OP3 R',     [  255    0  255]; ...   % Right Area OP3 (POperc)
      161, '7A L',      [    0  102    0]; ...   % Left Area 7A (SPL)
      162, '7A R',      [    0  102    0]; ...   % Right Area 7A (SPL)
      163, 'OP8 L',     [    0  204    0]; ...   % Left Area OP8 (Frontal Operculum)
      164, 'OP8 R',     [    0  204    0]; ...   % Right Area OP8 (Frontal Operculum)
      165, 'Subic L',   [  102  102  255]; ...   % Left Subiculum (Hippocampus)
      166, 'Subic R',   [  102  102  255]; ...   % Right Subiculum (Hippocampus)
      167, 'p29 L',     [    0  255  255]; ...   % Left Area p29 (retrosplenial)
      168, 'p29 R',     [    0  255  255]; ...   % Right Area p29 (retrosplenial)
      169, '25 L',      [  102    0    0]; ...   % Left Area 25 (sACC)
      170, '25 R',      [  102    0    0]; ...   % Right Area 25 (sACC)
      171, 'OP5 L',     [  255  215    0]; ...   % Left Area OP5 (Frontal Operculum)
      172, 'OP5 R',     [  255  215    0]; ...   % Right Area OP5 (Frontal Operculum)
      173, 'CA3 L',     [  255    0    0]; ...   % Left CA3 (Hippocampus)
      174, 'CA3 R',     [  255    0    0]; ...   % Right CA3 (Hippocampus)
      175, 'PFm L',     [  255  177  100]; ...   % Left Area PFm (IPL)
      176, 'PFm R',     [  255  177  100]; ...   % Right Area PFm (IPL)
      177, 'GapF-II L', [  255    0  255]; ...   % Left GapMap Frontal-II (GapMap)
      178, 'GapF-II R', [  255    0  255]; ...   % Right GapMap Frontal-II (GapMap)
      179, 'LB L',      [    0  102    0]; ...   % Left LB (Amygdala)
      180, 'LB R',      [    0  102    0]; ...   % Right LB (Amygdala)
      181, '4a L',      [    0  204    0]; ...   % Left Area 4a (PreCG)
      182, '4a R',      [    0  204    0]; ...   % Right Area 4a (PreCG)
      183, 'STS1 L',    [  102  102  255]; ...   % Left Area STS1 (STS)
      184, 'STS1 R',    [  102  102  255]; ...   % Right Area STS1 (STS)
      185, '5L L',      [    0  255  255]; ...   % Left Area 5L (SPL)
      186, '5L R',      [    0  255  255]; ...   % Right Area 5L (SPL)
      187, '5M L',      [  102    0    0]; ...   % Left Area 5M (SPL)
      188, '5M R',      [  102    0    0]; ...   % Right Area 5M (SPL)
      189, 'GapF-T L',  [  255  215    0]; ...   % Left GapMap Frontal-to-Temporal (GapMap)
      190, 'GapF-T R',  [  255  215    0]; ...   % Right GapMap Frontal-to-Temporal (GapMap)
      191, 'FG3 L',     [  255    0    0]; ...   % Left Area FG3 (FusG)
      192, 'FG3 R',     [  255    0    0]; ...   % Right Area FG3 (FusG)
      193, 'i30 L',     [  255  177  100]; ...   % Left Area i30 (retrosplenial)
      194, 'i30 R',     [  255  177  100]; ...   % Right Area i30 (retrosplenial)
      195, 'p30 L',     [  255    0  255]; ...   % Left Area p30 (retrosplenial)
      196, 'p30 R',     [  255    0  255]; ...   % Right Area p30 (retrosplenial)
      197, 'Fp1 L',     [    0  102    0]; ...   % Left Area Fp1 (FPole)
      198, 'Fp1 R',     [    0  102    0]; ...   % Right Area Fp1 (FPole)
      199, 'a30 L',     [    0  204    0]; ...   % Left Area a30 (retrosplenial)
      200, 'a30 R',     [    0  204    0]; ...   % Right Area a30 (retrosplenial)
      201, 'FN L',      [  102  102  255]; ...   % Left Fastigial Nucleus (Cerebellum)
      202, 'FN R',      [  102  102  255]; ...   % Right Fastigial Nucleus (Cerebellum)
      203, 's32 L',     [    0  255  255]; ...   % Left Area s32 (sACC)
      204, 's32 R',     [    0  255  255]; ...   % Right Area s32 (sACC)
      205, 'MF L',      [  102    0    0]; ...   % Left MF (Amygdala)
      206, 'MF R',      [  102    0    0]; ...   % Right MF (Amygdala)
      207, '3a L',      [  255  215    0]; ...   % Left Area 3a (PostCG)
      208, '3a R',      [  255  215    0]; ...   % Right Area 3a (PostCG)
      209, 'hOc5 L',    [  255    0    0]; ...   % Left Area hOc5 (LOC)
      210, 'hOc5 R',    [  255    0    0]; ...   % Right Area hOc5 (LOC)
      211, 'Id7 L',     [  255  177  100]; ...   % Left Area Id7 (Insula)
      212, 'Id7 R',     [  255  177  100]; ...   % Right Area Id7 (Insula)
      213, 'FG4 L',     [  255    0  255]; ...   % Left Area FG4 (FusG)
      214, 'FG4 R',     [  255    0  255]; ...   % Right Area FG4 (FusG)
      215, 'IN L',      [    0  102    0]; ...   % Left Interposed Nucleus (Cerebellum)
      216, 'IN R',      [    0  102    0]; ...   % Right Interposed Nucleus (Cerebellum)
      217, 'Fo1 L',     [    0  204    0]; ...   % Left Area Fo1 (OFC)
      218, 'Fo1 R',     [    0  204    0]; ...   % Right Area Fo1 (OFC)
      219, 'SF L',      [  102  102  255]; ...   % Left SF (Amygdala)
      220, 'SF R',      [  102  102  255]; ...   % Right SF (Amygdala)
      221, 's24 L',     [    0  255  255]; ...   % Left Area s24 (sACC)
      222, 's24 R',     [    0  255  255]; ...   % Right Area s24 (sACC)
      223, 'Fo6 L',     [  102    0    0]; ...   % Left Area Fo6 (OFC)
      224, 'Fo6 R',     [  102    0    0]; ...   % Right Area Fo6 (OFC)
      225, 'hOc4v L',   [  255  215    0]; ...   % Left Area hOc4v (LingG)
      226, 'hOc4v R',   [  255  215    0]; ...   % Right Area hOc4v (LingG)
      227, 'PFop L',    [  255    0    0]; ...   % Left Area PFop (IPL)
      228, 'PFop R',    [  255    0    0]; ...   % Right Area PFop (IPL)
      229, 'p32 L',     [  255  177  100]; ...   % Left Area p32 (pACC)
      230, 'p32 R',     [  255  177  100]; ...   % Right Area p32 (pACC)
      231, 'hOc1 L',    [  255    0  255]; ...   % Left Area hOc1 (V1, 17, CalcS)
      232, 'hOc1 R',    [  255    0  255]; ...   % Right Area hOc1 (V1, 17, CalcS)
      233, 'Fo2 L',     [    0  102    0]; ...   % Left Area Fo2 (OFC)
      234, 'Fo2 R',     [    0  102    0]; ...   % Right Area Fo2 (OFC)
      235, 'hIP7 L',    [    0  204    0]; ...   % Left Area hIP7 (IPS)
      236, 'hIP7 R',    [    0  204    0]; ...   % Right Area hIP7 (IPS)
      237, 'TE2.2 L',   [  102  102  255]; ...   % Left Area TE 2.2 (STG)
      238, 'TE2.2 R',   [  102  102  255]; ...   % Right Area TE 2.2 (STG)
      239, 'i29 L',     [    0  255  255]; ...   % Left Area i29 (retrosplenial)
      240, 'i29 R',     [    0  255  255]; ...   % Right Area i29 (retrosplenial)
      241, '7PC L',     [  102    0    0]; ...   % Left Area 7PC (SPL)
      242, '7PC R',     [  102    0    0]; ...   % Right Area 7PC (SPL)
      243, 'hOc4lp L',  [  255  215    0]; ...   % Left Area hOc4lp (LOC)
      244, 'hOc4lp R',  [  255  215    0]; ...   % Right Area hOc4lp (LOC)
      245, 'TeI L',     [  255    0    0]; ...   % Left Area TeI (STG)
      246, 'TeI R',     [  255    0    0]; ...   % Right Area TeI (STG)
      247, 'Fo3 L',     [  255  177  100]; ...   % Left Area Fo3 (OFC)
      248, 'Fo3 R',     [  255  177  100]; ...   % Right Area Fo3 (OFC)
};