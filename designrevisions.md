Go through a multistage ui plan to implement the following features and updates to the code

3d Spiderplot placement for parameter values, resnorm spectrum, and cube is not good (too small and constricted within the actual div-- needs to be a z value higher akin to the reset focus navigate and tooltip elemenets)

Redesign for settings menu; I have provided the changes on the settings bar (to the right of the visualization)
for the following visualizations that can be toggled from the top left

* Spider 2d:
* Spider 3d:
* Nyquist:

The ground truth toggle

Further changes on each visualization model will include:
* Spider 2d:
* Spider 3d:
* Nyquist:


everytime a change is made to a circuit, I do not want an entirely new circuit to be saved, it should just save the configuration changes to the same circuit. Also,   │
│   say I have clicked on a circuit and have it running in the playground, if I click on another circuit - not new circuit, it should still keep everything rendered in    │
│   or at least automatically render when I go back to it with a quick load time. Mainly I need to lower load times with the workers (keep improving) and also             │
│   drastically improve the speed and fluidity of the 3d model -                                                                                                           │
╰────────────────────────────────────────────────────────────────────