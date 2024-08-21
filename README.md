# Rumination_State_Task_DYSMARK
This experiment is to use psychopy to redo the rumination state task (RST).
 
RST consists of 4 conditions: resting state, sad autobiographical memory, rumination, 
and distraction.
 
The major rationale of the RST is to induce the participants into a self-constrained, 
continuous active rumination state.

The resting state is the baseline.

The sad autobiographical memory condition is for mood induction so that participants can 
ruminate more smoothly. The participants themselves generated the keywords before 
the MRI.

The rumination condition is of major interest.

The distraction condition serves as the control condition.

It is recommended that participants are briefed on the protocol and purpose of RST before 
undergo the actual scan.

To individualize the keywords of the dysphoric memory,
Make a .xlsx file in the following path: <RST directory>/dysphoric_memory_keywords/subject_{subj_ID}.xlsx
It is recommended to copy the existing <RST directory>/dysphoric_memory_keywords/subject_999.xlsx
and change the keywords. If participants provided less than four keywords for an event, you may
leave the rest of the columns blank. For example, if a given participant only provided one 
keyword for a dysphoric event, you may fill in the keyword1 column and leave the remaining three columns
blank.

Caution: in the filename (subject_{subj_ID}.xlsx) of this .xlsx, the {subj_ID} needs to be exactly the 
same as the subject ID you provided in the "participant" section at the beginning of the 
experiment. For example, if you labeled the present participant as "001", then named the .xlsx
file as subject_1.xlsx would not be accepted. Name it as subject_001.xlsx would do.

Xiao Chen
240430

The DYSMARK version:
Replace the original concrete distraction with the abstract distraction
chenxiaophd@gmail.com

References:
Jia, F. N.#, Chen, X.#, et al. (2023). 
Aberrant degree centrality profiles during rumination in major depressive disorder. 
Hum Brain Mapp, 44(17), 6245-6257.

Chen, X. & Yan, C. G.* (2021) Hypostability in the default mode network and hyperstability 
in the frontoparietal control network of dynamic functional architecture during rumination. 
NeuroImage, 2021, 118427.

Chen, X., Chen, N. X., Shen, Y. Q., Li, H. X., Li, L., Lu, B., ... & Yan, C. G.* (2020). 
The subsystem mechanism of default mode network underlying rumination: 
A reproducible neuroimaging study. NeuroImage, 221, 117185. 
