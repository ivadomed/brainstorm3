function Labels = mri_getlabels_aseg()
% ATLAS     : FreeSurfer ASEG + Desikan-Killiany (2006) + Destrieux (2010)
% REFERENCE : https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

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
    ... % ASEG
        0, 'Unknown',                        [    0    0    0]; ...
        1, 'Cortex ext L',                   [   70  130  180]; ...
        2, 'White L',                        [  245  245  245]; ...
        3, 'Cortex L',                       [  205   62   78]; ...
        4, 'Ventricle lat L',                [  120   18  134]; ...
        5, 'Ventricle inf-lat L',            [  196   58  250]; ...
        6, 'Cerebellum ext L',               [    0  148    0]; ...
        7, 'Cerebellum white L',             [  220  248  164]; ...
        8, 'Cerebellum L',                   [  230  148   34]; ...
        9, 'Thalamus L',                     [    0  118   14]; ...
       10, 'Thalamus L',                     [    0  118   14]; ...
       11, 'Caudate L',                      [  122  186  220]; ...
       12, 'Putamen L',                      [  236   13  176]; ...
       13, 'Pallidum L',                     [   12   48  255]; ...
       14, '3rd-Ventricle',                  [  204  182  142]; ...
       15, '4th-Ventricle',                  [   42  204  164]; ...
       16, 'Brainstem',                      [  119  159  176]; ...
       17, 'Hippocampus L',                  [  220  216   20]; ...
       18, 'Amygdala L',                     [  103  255  255]; ...
       19, 'Insula L',                       [   80  196   98]; ...
       20, 'Operculum L',                    [   60   58  210]; ...
       21, 'Line-1',                         [   60   58  210]; ...
       22, 'Line-2',                         [   60   58  210]; ...
       23, 'Line-3',                         [   60   58  210]; ...
       24, 'CSF',                            [   60   60   60]; ...
       25, 'Lesion L',                       [  255  165    0]; ...
       26, 'Accumbens L',                    [  255  165    0]; ...
       27, 'Substancia-Nigra L',             [    0  255  127]; ...
       28, 'VentralDC L',                    [  165   42   42]; ...
       29, 'Undetermined L',                 [  135  206  235]; ...
       30, 'Vessel L',                       [  160   32  240]; ...
       31, 'Choroid-plexus L',               [    0  200  200]; ...
       32, 'F3orb L',                        [  100   50  100]; ...
       33, 'lOg L',                          [  135   50   74]; ...
       34, 'aOg L',                          [  122  135   50]; ...
       35, 'mOg L',                          [   51   50  135]; ...
       36, 'pOg L',                          [   74  155   60]; ...
       37, 'Stellate L',                     [  120   62   43]; ...
       38, 'Porg L',                         [   74  155   60]; ...
       39, 'Aorg L',                         [  122  135   50]; ...
       40, 'Cortex ext R',                   [   70  130  180]; ...
       41, 'White R',                        [  245  245  245]; ...
       42, 'Cortex R',                       [  205   62   78]; ...
       43, 'Ventricle lat R',                [  120   18  134]; ...
       44, 'Ventricle inf-lat R',            [  196   58  250]; ...
       45, 'Cerebellum ext R',               [    0  148    0]; ...
       46, 'Cerebellum white R',             [  220  248  164]; ...
       47, 'Cerebellum R',                   [  230  148   34]; ...
       48, 'Thalamus R',                     [    0  118   14]; ...
       49, 'Thalamus R',                     [    0  118   14]; ...
       50, 'Caudate R',                      [  122  186  220]; ...
       51, 'Putamen R',                      [  236   13  176]; ...
       52, 'Pallidum R',                     [   13   48  255]; ...
       53, 'Hippocampus R',                  [  220  216   20]; ...
       54, 'Amygdala R',                     [  103  255  255]; ...
       55, 'Insula R',                       [   80  196   98]; ...
       56, 'Operculum R',                    [   60   58  210]; ...
       57, 'Lesion R',                       [  255  165    0]; ...
       58, 'Accumbens R',                    [  255  165    0]; ...
       59, 'Substancia-Nigra R',             [    0  255  127]; ...
       60, 'VentralDC R',                    [  165   42   42]; ...
       61, 'Undetermined R',                 [  135  206  235]; ...
       62, 'Vessel R',                       [  160   32  240]; ...
       63, 'Choroid-plexus R',               [    0  200  221]; ...
       64, 'F3orb R',                        [  100   50  100]; ...
       65, 'lOg R',                          [  135   50   74]; ...
       66, 'aOg R',                          [  122  135   50]; ...
       67, 'mOg R',                          [   51   50  135]; ...
       68, 'pOg R',                          [   74  155   60]; ...
       69, 'Stellate R',                     [  120   62   43]; ...
       70, 'Porg R',                         [   74  155   60]; ...
       71, 'Aorg R',                         [  122  135   50]; ...
       72, '5th-Ventricle',                  [  120  190  150]; ...
       73, 'Interior L',                     [  122  135   50]; ...
       74, 'Interior R',                     [  122  135   50]; ...
       77, 'WM-hypointensities',             [  200   70  255]; ...
       78, 'WM-hypointensities L',           [  255  148   10]; ...
       79, 'WM-hypointensities R',           [  255  148   10]; ...
       80, 'non-WM-hypointensities',         [  164  108  226]; ...
       81, 'non-WM-hypointensities L',       [  164  108  226]; ...
       82, 'non-WM-hypointensities R',       [  164  108  226]; ...
       83, 'F1 L',                           [  255  218  185]; ...
       84, 'F1 R',                           [  255  218  185]; ...
       85, 'Optic-Chiasm',                   [  234  169   30]; ...
       96, 'Amygdala ant L',                 [  205   10  125]; ...
       97, 'Amygdala ant R',                 [  205   10  125]; ...
       98, 'Dura',                           [  160   32  240]; ...
      100, 'WM-abnormal L',                  [  124  140  178]; ...
      101, 'Caudate-abnormal L',             [  125  140  178]; ...
      102, 'Putamen-abnormal L',             [  126  140  178]; ...
      103, 'Accumbens-abnormal L',           [  127  140  178]; ...
      104, 'Pallidum-abnormal L',            [  124  141  178]; ...
      105, 'Amygdala-abnormal L',            [  124  142  178]; ...
      106, 'Hippocampus-abnormal L',         [  124  143  178]; ...
      107, 'Thalamus-abnormal L',            [  124  144  178]; ...
      108, 'VDC-intensity-abnormality L',    [  124  140  179]; ...
      109, 'WM-abnormal R',                  [  124  140  178]; ...
      110, 'Caudate-abnormal R',             [  125  140  178]; ...
      111, 'Putamen-abnormal R',             [  126  140  178]; ...
      112, 'Accumbens-abnormal R',           [  127  140  178]; ...
      113, 'Pallidum-abnormal R',            [  124  141  178]; ...
      114, 'Amygdala-abnormal R',            [  124  142  178]; ...
      115, 'Hippocampus-abnormal R',         [  124  143  178]; ...
      116, 'Thalamus-abnormal R',            [  124  144  178]; ...
      117, 'VDC-abnormal R',                 [  124  140  179]; ...
      118, 'Epidermis',                      [  255   20  147]; ...
      119, 'Conn-Tissue',                    [  205  179  139]; ...
      120, 'SC-Fat-Muscle',                  [  238  238  209]; ...
      121, 'Cranium',                        [  200  200  200]; ...
      122, 'CSF-SA',                         [   74  255   74]; ...
      123, 'Muscle',                         [  238    0    0]; ...
      124, 'Ear',                            [    0    0  139]; ...
      125, 'Adipose',                        [  173  255   47]; ...
      126, 'Spinal-Cord',                    [  133  203  229]; ...
      127, 'Soft-Tissue',                    [   26  237   57]; ...
      128, 'Nerve',                          [   34  139   34]; ...
      129, 'Bone',                           [   30  144  255]; ...
      130, 'Air',                            [  147   19  173]; ...
      131, 'Orbital-Fat',                    [  238   59   59]; ...
      132, 'Tongue',                         [  221   39  200]; ...
      133, 'Nasal-Structures',               [  238  174  238]; ...
      134, 'Globe',                          [  255    0    0]; ...
      135, 'Teeth',                          [   72   61  139]; ...
      136, 'Caudate-Putamen L',              [   21   39  132]; ...
      137, 'Caudate-Putamen R',              [   21   39  132]; ...
      138, 'Claustrum L',                    [   65  135   20]; ...
      139, 'Claustrum R',                    [   65  135   20]; ...
      140, 'Cornea',                         [  134    4  160]; ...
      142, 'Diploe',                         [  221  226   68]; ...
      143, 'Vitreous-Humor',                 [  255  255  254]; ...
      144, 'Lens',                           [   52  209  226]; ...
      145, 'Aqueous-Humor',                  [  239  160  223]; ...
      146, 'Outer-Table',                    [   70  130  180]; ...
      147, 'Inner-Table',                    [   70  130  181]; ...
      148, 'Periosteum',                     [  139  121   94]; ...
      149, 'Endosteum',                      [  224  224  224]; ...
      150, 'R-C-S',                          [  255    0    0]; ...
      151, 'Iris',                           [  205  205    0]; ...
      152, 'SC-Adipose-Muscle',              [  238  238  209]; ...
      153, 'SC-Tissue',                      [  139  121   94]; ...
      154, 'Orbital-Adipose',                [  238   59   59]; ...
      155, 'IntCapsule-Ant L',               [  238   59   59]; ...
      156, 'IntCapsule-Ant R',               [  238   59   59]; ...
      157, 'IntCapsule-Pos L',               [   62   10  205]; ...
      158, 'IntCapsule-Pos R',               [   62   10  205]; ...
   ... % These labels are for babies/children
      159, 'Cerebral-WM-unmyelinated L',     [    0  118   14]; ...
      160, 'Cerebral-WM-unmyelinated R',     [    0  118   14]; ...
      161, 'Cerebral-WM-myelinated L',       [  220  216   21]; ...
      162, 'Cerebral-WM-myelinated R',       [  220  216   21]; ...
      163, 'Subcortical-Gray-Matter L',      [  122  186  220]; ...
      164, 'Subcortical-Gray-Matter R',      [  122  186  220]; ...
      165, 'Skull',                          [  120  120  120]; ...
      166, 'Posterior-fossa',                [   14   48  255]; ...
      167, 'Scalp',                          [  166   42   42]; ...
      168, 'Hematoma',                       [  121   18  134]; ...
      169, 'Basal-Ganglia L',                [  236   13  127]; ...
      176, 'Basal-Ganglia R',                [  236   13  126]; ...
   ... % Label names and colors for Brainstem consituents
      170, 'Brainstem',                      [  119  159  176]; ...
      171, 'DCG',                            [  119    0  176]; ...
      172, 'Vermis',                         [  119  100  176]; ...
      173, 'Midbrain',                       [  242  104   76]; ...
      174, 'Pons',                           [  206  195   58]; ...
      175, 'Medulla',                        [  119  159  176]; ...
      177, 'Vermis-White-Matter',            [  119   50  176]; ...
      178, 'SCP',                            [  142  182    0]; ...
      179, 'Floculus',                       [   19  100  176]; ...
      180, 'Cortical-Dysplasia L',           [   73   61  139]; ...
      181, 'Cortical-Dysplasia R',           [   73   62  139]; ...
      182, 'CblumNodulus',                   [   10  100  176]; ...
      193, 'Hippocampal_fissure L',          [    0  196  255]; ...
      194, 'CADG-head L',                    [  255  164  164]; ...
      195, 'Subiculum L',                    [  196  196    0]; ...
      196, 'Fimbria L',                      [    0  100  255]; ...
      197, 'Hippocampal_fissure R',          [  128  196  164]; ...
      198, 'CADG-head R',                    [    0  126   75]; ...
      199, 'Subiculum R',                    [  128   96   64]; ...
      200, 'Fimbria R',                      [    0   50  128]; ...
      201, 'Alveus',                         [  255  204  153]; ...
      202, 'Perforant_pathway',              [  255  128  128]; ...
      203, 'Parasubiculum',                  [  255  255    0]; ...
      204, 'Presubiculum',                   [   64    0   64]; ...
      205, 'Subiculum',                      [    0    0  255]; ...
      206, 'CA1',                            [  255    0    0]; ...
      207, 'CA2',                            [  128  128  255]; ...
      208, 'CA3',                            [    0  128    0]; ...
      209, 'CA4',                            [  196  160  128]; ...
      210, 'GC-DG',                          [   32  200  255]; ...
      211, 'HATA',                           [  128  255  128]; ...
      212, 'Fimbria',                        [  204  153  204]; ...
      213, 'Lateral_ventricle',              [  121   17  136]; ...
      214, 'Molecular_layer_HP',             [  128    0    0]; ...
      215, 'Hippocampal_fissure',            [  128   32  255]; ...
      216, 'Entorhinal_cortex',              [  255  204  102]; ...
      217, 'Molecular_layer_subiculum',      [  128  128  128]; ...
      218, 'Amygdala',                       [  104  255  255]; ...
      219, 'Cerebral_White_Matter',          [    0  226    0]; ...
      220, 'Cerebral_Cortex',                [  205   63   78]; ...
      221, 'Inf_Lat_Vent',                   [  197   58  250]; ...
      222, 'Perirhinal',                     [   33  150  250]; ...
      223, 'Cerebral_White_Matter_Edge',     [  226    0    0]; ...
      224, 'Background',                     [  100  100  100]; ...
      225, 'Ectorhinal',                     [  197  150  250]; ...
      226, 'HP_tail',                        [  170  170  255]; ...
      250, 'Fornix',                         [  255    0    0]; ...
      251, 'CC_Posterior',                   [    0    0   64]; ...
      252, 'CC_Mid_Posterior',               [    0    0  112]; ...
      253, 'CC_Central',                     [    0    0  160]; ...
      254, 'CC_Mid_Anterior',                [    0    0  208]; ...
      255, 'CC_Anterior',                    [    0    0  255]; ...
   ... % aparc
     1000, 'Unknown L',                      [   25    5   25]; ...
     1001, 'Banks sts L',                    [   25  100   40]; ...
     1002, 'Caudal ant cing L',              [  125  100  160]; ...
     1003, 'Caudal mid front L',             [  100   25    0]; ...
     1004, 'Corpus callosum L',              [  120   70   50]; ...
     1005, 'Cuneus L',                       [  220   20  100]; ...
     1006, 'Entorhinal L',                   [  220   20   10]; ...
     1007, 'Fusiform L',                     [  180  220  140]; ...
     1008, 'Inferior parietal L',            [  220   60  220]; ...
     1009, 'Inferior temporal L',            [  180   40  120]; ...
     1010, 'Isthmus cingulate L',            [  140   20  140]; ...
     1011, 'Lateral occipital L',            [   20   30  140]; ...
     1012, 'Lateral orbitofrontal L',        [   35   75   50]; ...
     1013, 'Lingual L',                      [  225  140  140]; ...
     1014, 'Medial orbitofrontal L',         [  200   35   75]; ...
     1015, 'Middle temporal L',              [  160  100   50]; ...
     1016, 'Parahippocampal L',              [   20  220   60]; ...
     1017, 'Paracentral L',                  [   60  220   60]; ...
     1018, 'Pars opercularis L',             [  220  180  140]; ...
     1019, 'Pars orbitalis L',               [   20  100   50]; ...
     1020, 'Pars triangularis L',            [  220   60   20]; ...
     1021, 'Pericalcarine L',                [  120  100   60]; ...
     1022, 'Postcentral L',                  [  220   20   20]; ...
     1023, 'Posterior cingulate L',          [  220  180  220]; ...
     1024, 'Precentral L',                   [   60   20  220]; ...
     1025, 'Precuneus L',                    [  160  140  180]; ...
     1026, 'Rostral ant cing L',             [   80   20  140]; ...
     1027, 'Rostral mid front L',            [   75   50  125]; ...
     1028, 'Superior frontal L',             [   20  220  160]; ...
     1029, 'Superior parietal L',            [   20  180  140]; ...
     1030, 'Superior temporal L',            [  140  220  220]; ...
     1031, 'Supramarginal L',                [   80  160   20]; ...
     1032, 'Frontal pole L',                 [  100    0  100]; ...
     1033, 'Temporal pole L',                [   70   70   70]; ...
     1034, 'Transverse temporal L',          [  150  150  200]; ...
     1035, 'Insula L',                       [  255  192   32]; ...
     2000, 'Unknown R',                      [   25    5   25]; ...
     2001, 'Banks sts R',                    [   25  100   40]; ...
     2002, 'Caudal ant cing R',              [  125  100  160]; ...
     2003, 'Caudal mid front R',             [  100   25    0]; ...
     2004, 'Corpus callosum R',              [  120   70   50]; ...
     2005, 'Cuneus R',                       [  220   20  100]; ...
     2006, 'Entorhinal R',                   [  220   20   10]; ...
     2007, 'Fusiform R',                     [  180  220  140]; ...
     2008, 'Inferior parietal R',            [  220   60  220]; ...
     2009, 'Inferior temporal R',            [  180   40  120]; ...
     2010, 'Isthmus cingulate R',            [  140   20  140]; ...
     2011, 'Lateral occipital R',            [   20   30  140]; ...
     2012, 'Lateral orbitofrontal R',        [   35   75   50]; ...
     2013, 'Lingual R',                      [  225  140  140]; ...
     2014, 'Medial orbitofrontal R',         [  200   35   75]; ...
     2015, 'Middle temporal R',              [  160  100   50]; ...
     2016, 'Parahippocampal R',              [   20  220   60]; ...
     2017, 'Paracentral R',                  [   60  220   60]; ...
     2018, 'Pars opercularis R',             [  220  180  140]; ...
     2019, 'Pars orbitalis R',               [   20  100   50]; ...
     2020, 'Pars triangularis R',            [  220   60   20]; ...
     2021, 'Pericalcarine R',                [  120  100   60]; ...
     2022, 'Postcentral R',                  [  220   20   20]; ...
     2023, 'Posterior cingulate R',          [  220  180  220]; ...
     2024, 'Precentral R',                   [   60   20  220]; ...
     2025, 'Precuneus R',                    [  160  140  180]; ...
     2026, 'Rostral ant cing R',             [   80   20  140]; ...
     2027, 'Rostral mid front R',            [   75   50  125]; ...
     2028, 'Superior frontal R',             [   20  220  160]; ...
     2029, 'Superior parietal R',            [   20  180  140]; ...
     2030, 'Superior temporal R',            [  140  220  220]; ...
     2031, 'Supramarginal R',                [   80  160   20]; ...
     2032, 'Frontal pole R',                 [  100    0  100]; ...
     2033, 'Temporal pole R',                [   70   70   70]; ...
     2034, 'Transverse temporal R',          [  150  150  200]; ...
     2035, 'Insula R',                       [  255  192   32]; ...
  ... % aparc.a2009s
    11100, 'Unknown L',                      [    0    0    0]; ...
    11101, 'G_and_S_frontomargin L',         [   23  220   60]; ...
    11102, 'G_and_S_occipital_inf L',        [   23   60  180]; ...
    11103, 'G_and_S_paracentral L',          [   63  100   60]; ...
    11104, 'G_and_S_subcentral L',           [   63   20  220]; ...
    11105, 'G_and_S_transv_frontopol L',     [   13    0  250]; ...
    11106, 'G_and_S_cingul-Ant L',           [   26   60    0]; ...
    11107, 'G_and_S_cingul-Mid-Ant L',       [   26   60   75]; ...
    11108, 'G_and_S_cingul-Mid-Post L',      [   26   60  150]; ...
    11109, 'G_cingul-Post-dorsal L',         [   25   60  250]; ...
    11110, 'G_cingul-Post-ventral L',        [   60   25   25]; ...
    11111, 'G_cuneus L',                     [  180   20   20]; ...
    11112, 'G_front_inf-Opercular L',        [  220   20  100]; ...
    11113, 'G_front_inf-Orbital L',          [  140   60   60]; ...
    11114, 'G_front_inf-Triangul L',         [  180  220  140]; ...
    11115, 'G_front_middle L',               [  140  100  180]; ...
    11116, 'G_front_sup L',                  [  180   20  140]; ...
    11117, 'G_Ins_lg_and_S_cent_ins L',      [   23   10   10]; ...
    11118, 'G_insular_short L',              [  225  140  140]; ...
    11119, 'G_occipital_middle L',           [  180   60  180]; ...
    11120, 'G_occipital_sup L',              [   20  220   60]; ...
    11121, 'G_oc-temp_lat-fusifor L',        [   60   20  140]; ...
    11122, 'G_oc-temp_med-Lingual L',        [  220  180  140]; ...
    11123, 'G_oc-temp_med-Parahip L',        [   65  100   20]; ...
    11124, 'G_orbital L',                    [  220   60   20]; ...
    11125, 'G_pariet_inf-Angular L',         [   20   60  220]; ...
    11126, 'G_pariet_inf-Supramar L',        [  100  100   60]; ...
    11127, 'G_parietal_sup L',               [  220  180  220]; ...
    11128, 'G_postcentral L',                [   20  180  140]; ...
    11129, 'G_precentral L',                 [   60  140  180]; ...
    11130, 'G_precuneus L',                  [   25   20  140]; ...
    11131, 'G_rectus L',                     [   20   60  100]; ...
    11132, 'G_subcallosal L',                [   60  220   20]; ...
    11133, 'G_temp_sup-G_T_transv L',        [   60   60  220]; ...
    11134, 'G_temp_sup-Lateral L',           [  220   60  220]; ...
    11135, 'G_temp_sup-Plan_polar L',        [   65  220   60]; ...
    11136, 'G_temp_sup-Plan_tempo L',        [   25  140   20]; ...
    11137, 'G_temporal_inf L',               [  220  220  100]; ...
    11138, 'G_temporal_middle L',            [  180   60   60]; ...
    11139, 'Lat_Fis-ant-Horizont L',         [   61   20  220]; ...
    11140, 'Lat_Fis-ant-Vertical L',         [   61   20   60]; ...
    11141, 'Lat_Fis-post L',                 [   61   60  100]; ...
    11142, 'Medial_wall L',                  [   25   25   25]; ...
    11143, 'Pole_occipital L',               [  140   20   60]; ...
    11144, 'Pole_temporal L',                [  220  180   20]; ...
    11145, 'S_calcarine L',                  [   63  180  180]; ...
    11146, 'S_central L',                    [  221   20   10]; ...
    11147, 'S_cingul-Marginalis L',          [  221   20  100]; ...
    11148, 'S_circular_insula_ant L',        [  221   60  140]; ...
    11149, 'S_circular_insula_inf L',        [  221   20  220]; ...
    11150, 'S_circular_insula_sup L',        [   61  220  220]; ...
    11151, 'S_collat_transv_ant L',          [  100  200  200]; ...
    11152, 'S_collat_transv_post L',         [   10  200  200]; ...
    11153, 'S_front_inf L',                  [  221  220   20]; ...
    11154, 'S_front_middle L',               [  141   20  100]; ...
    11155, 'S_front_sup L',                  [   61  220  100]; ...
    11156, 'S_interm_prim-Jensen L',         [  141   60   20]; ...
    11157, 'S_intrapariet_and_P_trans L',    [  143   20  220]; ...
    11158, 'S_oc_middle_and_Lunatus L',      [  101   60  220]; ...
    11159, 'S_oc_sup_and_transversal L',     [   21   20  140]; ...
    11160, 'S_occipital_ant L',              [   61   20  180]; ...
    11161, 'S_oc-temp_lat L',                [  221  140   20]; ...
    11162, 'S_oc-temp_med_and_Lingual L',    [  141  100  220]; ...
    11163, 'S_orbital_lateral L',            [  221  100   20]; ...
    11164, 'S_orbital_med-olfact L',         [  181  200   20]; ...
    11165, 'S_orbital-H_Shaped L',           [  101   20   20]; ...
    11166, 'S_parieto_occipital L',          [  101  100  180]; ...
    11167, 'S_pericallosal L',               [  181  220   20]; ...
    11168, 'S_postcentral L',                [   21  140  200]; ...
    11169, 'S_precentral-inf-part L',        [   21   20  240]; ...
    11170, 'S_precentral-sup-part L',        [   21   20  200]; ...
    11171, 'S_suborbital L',                 [   21   20   60]; ...
    11172, 'S_subparietal L',                [  101   60   60]; ...
    11173, 'S_temporal_inf L',               [   21  180  180]; ...
    11174, 'S_temporal_sup L',               [  223  220   60]; ...
    11175, 'S_temporal_transverse L',        [  221   60   60]; ...
    12100, 'Unknown R',                      [    0    0    0]; ...
    12101, 'G_and_S_frontomargin R',         [   23  220   60]; ...
    12102, 'G_and_S_occipital_inf R',        [   23   60  180]; ...
    12103, 'G_and_S_paracentral R',          [   63  100   60]; ...
    12104, 'G_and_S_subcentral R',           [   63   20  220]; ...
    12105, 'G_and_S_transv_frontopol R',     [   13    0  250]; ...
    12106, 'G_and_S_cingul-Ant R',           [   26   60    0]; ...
    12107, 'G_and_S_cingul-Mid-Ant R',       [   26   60   75]; ...
    12108, 'G_and_S_cingul-Mid-Post R',      [   26   60  150]; ...
    12109, 'G_cingul-Post-dorsal R',         [   25   60  250]; ...
    12110, 'G_cingul-Post-ventral R',        [   60   25   25]; ...
    12111, 'G_cuneus R',                     [  180   20   20]; ...
    12112, 'G_front_inf-Opercular R',        [  220   20  100]; ...
    12113, 'G_front_inf-Orbital R',          [  140   60   60]; ...
    12114, 'G_front_inf-Triangul R',         [  180  220  140]; ...
    12115, 'G_front_middle R',               [  140  100  180]; ...
    12116, 'G_front_sup R',                  [  180   20  140]; ...
    12117, 'G_Ins_lg_and_S_cent_ins R',      [   23   10   10]; ...
    12118, 'G_insular_short R',              [  225  140  140]; ...
    12119, 'G_occipital_middle R',           [  180   60  180]; ...
    12120, 'G_occipital_sup R',              [   20  220   60]; ...
    12121, 'G_oc-temp_lat-fusifor R',        [   60   20  140]; ...
    12122, 'G_oc-temp_med-Lingual R',        [  220  180  140]; ...
    12123, 'G_oc-temp_med-Parahip R',        [   65  100   20]; ...
    12124, 'G_orbital R',                    [  220   60   20]; ...
    12125, 'G_pariet_inf-Angular R',         [   20   60  220]; ...
    12126, 'G_pariet_inf-Supramar R',        [  100  100   60]; ...
    12127, 'G_parietal_sup R',               [  220  180  220]; ...
    12128, 'G_postcentral R',                [   20  180  140]; ...
    12129, 'G_precentral R',                 [   60  140  180]; ...
    12130, 'G_precuneus R',                  [   25   20  140]; ...
    12131, 'G_rectus R',                     [   20   60  100]; ...
    12132, 'G_subcallosal R',                [   60  220   20]; ...
    12133, 'G_temp_sup-G_T_transv R',        [   60   60  220]; ...
    12134, 'G_temp_sup-Lateral R',           [  220   60  220]; ...
    12135, 'G_temp_sup-Plan_polar R',        [   65  220   60]; ...
    12136, 'G_temp_sup-Plan_tempo R',        [   25  140   20]; ...
    12137, 'G_temporal_inf R',               [  220  220  100]; ...
    12138, 'G_temporal_middle R',            [  180   60   60]; ...
    12139, 'Lat_Fis-ant-Horizont R',         [   61   20  220]; ...
    12140, 'Lat_Fis-ant-Vertical R',         [   61   20   60]; ...
    12141, 'Lat_Fis-post R',                 [   61   60  100]; ...
    12142, 'Medial_wall R',                  [   25   25   25]; ...
    12143, 'Pole_occipital R',               [  140   20   60]; ...
    12144, 'Pole_temporal R',                [  220  180   20]; ...
    12145, 'S_calcarine R',                  [   63  180  180]; ...
    12146, 'S_central R',                    [  221   20   10]; ...
    12147, 'S_cingul-Marginalis R',          [  221   20  100]; ...
    12148, 'S_circular_insula_ant R',        [  221   60  140]; ...
    12149, 'S_circular_insula_inf R',        [  221   20  220]; ...
    12150, 'S_circular_insula_sup R',        [   61  220  220]; ...
    12151, 'S_collat_transv_ant R',          [  100  200  200]; ...
    12152, 'S_collat_transv_post R',         [   10  200  200]; ...
    12153, 'S_front_inf R',                  [  221  220   20]; ...
    12154, 'S_front_middle R',               [  141   20  100]; ...
    12155, 'S_front_sup R',                  [   61  220  100]; ...
    12156, 'S_interm_prim-Jensen R',         [  141   60   20]; ...
    12157, 'S_intrapariet_and_P_trans R',    [  143   20  220]; ...
    12158, 'S_oc_middle_and_Lunatus R',      [  101   60  220]; ...
    12159, 'S_oc_sup_and_transversal R',     [   21   20  140]; ...
    12160, 'S_occipital_ant R',              [   61   20  180]; ...
    12161, 'S_oc-temp_lat R',                [  221  140   20]; ...
    12162, 'S_oc-temp_med_and_Lingual R',    [  141  100  220]; ...
    12163, 'S_orbital_lateral R',            [  221  100   20]; ...
    12164, 'S_orbital_med-olfact R',         [  181  200   20]; ...
    12165, 'S_orbital-H_Shaped R',           [  101   20   20]; ...
    12166, 'S_parieto_occipital R',          [  101  100  180]; ...
    12167, 'S_pericallosal R',               [  181  220   20]; ...
    12168, 'S_postcentral R',                [   21  140  200]; ...
    12169, 'S_precentral-inf-part R',        [   21   20  240]; ...
    12170, 'S_precentral-sup-part R',        [   21   20  200]; ...
    12171, 'S_suborbital R',                 [   21   20   60]; ...
    12172, 'S_subparietal R',                [  101   60   60]; ...
    12173, 'S_temporal_inf R',               [   21  180  180]; ...
    12174, 'S_temporal_sup R',               [  223  220   60]; ...
    12175, 'S_temporal_transverse R',        [  221   60   60]; ...
};
