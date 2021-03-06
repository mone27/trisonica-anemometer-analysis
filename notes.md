20/11/2019
started working on importing data from TRS and WM

21/12/2019
you must invert the u of TRS1
v of TRS1 is perfect
w of TRS1 is not that good

23/12/2019
found correct definition of direction:
for WM and EddyPro:
u is positive when the wind flows from south to north
v is positive when the wind flows from east to west
w is positive when the wind goes up
here more info at page 7 http://gillinstruments.com/data/manuals/1561-PS-0001%20WindMaster%20Windmaster%20Pro%20Manual%20Issue%2015.pdf
For TRS (indefault setting called "otms", see page 7 https://anemoment.com/wp-content/uploads/2019/04/Trisonica-Mini-User-Manual-Ap2019.pdf)
u is positive when the wind flows from north to south
v is positive when the wind flows from east to west
w is positive when the wind goes up

This mean you need to invert u component of TRS

24/12/2018
rotating u and v for TRS2 by +/- 45° seems is not working to get correct axes coordinates
cannot understand how EddyPro is calculating wind_dir
wind_dir of TRS1 is currently wrong since we u has not been inverted

proposed correction for TRS2 (check them!)
u = - rot_45°(u)
v = -w
w = rot_45°(v)

29/12/2018
generated dataset with fixed axes for TRS1 using data_prepocessing.py.
For all data from August and Septmber (~12GB) this is the time it took:
`CPU times: user 6min 51s, sys: 15.5 s, total: 7min 6s`

15/01
trying to understand how EP calculates wind dir, need to understand why it adds pi in AngularAverageNoError (line 198)
Discovered!! pi is due to the fact tha atan2 returns from -180 to 180 and adding 180 seems it is equal to converting to make `a = a % 360` and then do `180-a`.

23/01
when you need to calc the average wind_speed you cannot calc the wind_speed for each observation and then average but you need to average the components and the calc the wind_speed :)



##### 2020 version ######

### 22/07/2020

configured M506 and M507 for outputting at 10Hz and with many paramas (see screens)
analyzed last ~2 weeks of WM data and seen most of the wind from 250 degrees
Installed for tests m506 (vertical, north aligned ??) m507 (horizontal), rotated 45 degres main axis aligned with main structure (320 degrees??, problemns with compass measurements)

### 23/07/2020
First analysis of the wind of the previous night, m506 u axis has been inverted (so to match EP refence system) and then processed with EP at 5 mins. The wind speed (calulated by EP) looks comparable, the wind direction (calculated by EP) completely inverted but looking at the individual components they were completely wrong. Tried also with M507 but got pretty bad results as well, no time to investigate it more.

Reinstallation of the instruments in the field:
- WM 1 and 2 have been slightly rotated on their base to better access wind from 250°, North OFFSET: 310°
- M506 has been slightly rotated, North Offset: 0°
- M507 has been installed on a rotated support being with the bottom (of the anemometer) facing 320° N, then has been also rotated on the pipe so that is you look at the top of the anemometer the north arrow is pointingon the bottom left ( 275° clockswise from the vertical)

Compass measueremenets has been quite difficult so the anemometers can have a +/- 10 ° error in their orientation

Measurement of the distance between anemometers center

M6 risp wm1
-35 cm  E
-7  cm vert
-5 cm N

M7 risp wm1

-60 cm E
-10 cm N
-7 cm vert

Wm2 risp wm1

33 cm E
-4 cm N
0 cm vert

Configuration:
WM1 com1
WM2 com2
m506 com3
m507 com4

Succesfully connected to wifi and teamviewer
Launched scanemone for recording data

Cheking in the evening m507 on com4 started but never worked, trying to force close the scanemone process and to reboot the computer but it is not working


### 24/07/2020

Forcing rebooting computer but still issues with reading on scanemone on com4 even it was working fine with putty on serial port

Swapping the port of WM1 with m507, so configuration is:
WM1 com1
WM2 com4
m506 com3
m507 com2

cheking that all anemometers are correctly working with putty, lauching scanemone.
First good data from 13:00 

WM2 on com4 works partially for some hours saves data other times it blocked


### 25/07/2020

Anemometers are logging correctly com4 sometimes is not working.

Trying to understand angle rotation 


this function is not working (don't ask why)
```python
def rotate(tens, ang, axes):
    rot_matrix = np.array([[np.cos(ang), -np.sin(ang)],
                       [np.sin(ang), np.cos(ang)]])
    return np.matmul(tens[:, axes], rot_matrix)
    
```

this is working (correctly rotating wind)

```python
def rotate_ang(df, ang):
    wind_dir, wind_speed = cart2pol(df.u, df.v)
    wind_dir += np.deg2rad(ang)
    return np.column_stack(pol2cart(wind_dir, wind_speed))
```

For wind dir found this website that explains it quite well http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv and this one https://www.eol.ucar.edu/content/wind-direction-quick-reference


### 27/07/2020

Trying to use inverted v for the m506 since the direction is exactly at the opposite, but not too sure it is going to work

### 28/07/2020

Created 3d model to work understand the necessaray tranformation for m507

Updated data_preprocessing and started analysis



### 29/07/2020

eddypro_TRS_m507_full_output_2020-07-29T075333_exp -> rotation of 45 
eddypro_TRS_m507_full_output_2020-07-28T163413_exp -> rotation of -45 (seems to not be working)

with this transforms components are still completely wrong, but overall wind speed looks promising


Looking at the angle needed for the rotations (check wm1_axis_rotation.svg) and coming to this conclusion:
You have reference system A (where you want end results) and you have reference system B (the one of input data)
From an A point of view B is rotated by alpha angle, to tranform any vector in B in A you need to rotate the vector by -alpha.
**NB** angles measured in _wind_ mode are the opposite of angles measured in _math_ mode 

for wm1(2) the transform needed is ```rotate_ang(wm1, 310)```   

Inverted v of M507 and seems it is much better


### 30/07/2020

Finalized (m506_analysis)[m506_analysis] 

Overall this is what is seems what needs to be done:

- Invert the u
- Invert the v
- Rotate by 27° N


#### Working on M507

recalculating needed rotations taking into account the inverted v

steps are 
1. rotate by 45 °
2. remap axis   
   u = -u  
   v = w   
   w = v   
   

When processing M507 with possible correct rotation 5-6 EP gives this error:
```Error(59)> At least one wind component appears to be corrupted (too many implausible values).
Error(59)> This may also be the result of data exclusion by the "Absolute limits" test or by a
Error(59)> custom-designed "Flag" in the "Basic Settings" page.
Error(59)> If the problem occurs for many or all raw files, check those settings.
```


### 31/07/2020

Trying to get the rotation right, rotations done with a reasoning behind are completely wrong but the fact the wind speed looks great give hopes.

Tried different methods to get by brute force the most efficient trasnsformation:
 1. rotation matrix with pytorch (gradient works fine but is an overkill and then the rotation matrix is difficult to interpret)
 2. Use scipy.transform.rotation the great news is that supports Euler angles (even if not so easy too understand)
 3. optimization with scipy minimize
 4. Brute force optimization: trying all the possible triplets of angles with ranges of 10°
 5. rotation.align_vectors !! (should be the fastest and easiest method)
 
 Obtained some rotations that make a sensible output ([240, 330, 90] in xyz):
 - u is around 0 most time, but when is bigger follows decently the wm1
 - v is almost perfect
 - w does follow the wm1 but is far from being good
 
 Then tried hard to visualize the obtained rotation, but thinkercard does not rotate on custom plane so at the end using rhinoceros which is way complex but this his job, however the result was pretty useless.
 Trying also 3d plotting from matplotlib, which probably is the best 
 Almost giving up :(
 way forward and the visualization is still useless since the rotation appears on random planes.
 
 Ideas of stuff that needs to be done:
 - understand why v is so good
 - ensure for the last time the reasoned transformations sucks
 - check is the optimized rotation is consistent across different days
 
 
 
 ### 01/08/2020
 
 **NEED TO WRITE**
 
 Checking that the data using the rotation from the optimization acutally looks good and filtering with wind is even better
 
 Created set of funcs to plot vectors in 3d and using rotation by step
 
 
### 02/08/2020
Difference between intrisic and estrinsic angles in euler rotations (I am stupid that I don' read properly ALL the documentation)

the magic angles found were for wm1 not wm1r!!!!

both [ 70., 200., 140.] and 250., 340., 320. looks like good rotation on XYZ with m7 compared to wm1 but life still sucks



verified that using process m_507 is the same thing of this rotation!!!

ang = [  -90,    0., -135]
seq='XYZ' 
 
#### The only sure thing

the theory rotation are correct in the 3D space but the data is **TOTALLY TERRIBLE**

using brute force we can obtain data that looks decent (not good) but they have no sense nor consistency


### 03/08/2020

Worked on notebook to summarize the situation of M507, discovered that without an inverted v it looks decent
and with a 30 ° rotation (instead of the theory of 45) starts to look decent, even though when filtered it starts to be worse

Conclusion is that M507 is quite bad at measuring w, and virtually all roads to make something sensible out of it has been explored.

Downloaded new data from the sensors.

### 09/08/2020

First test to compare fluxes, Tau can be okay but there is an high difference in H due to big differences in temperatures.

Trying to filter wind dir of m507 at high frequency by replacing other values with -9999. EP understand this notation, but the result is still not equal to the EP Wind Filter function... WIP

Checked windrose and it seems m507 is not taking the biggest amount of wind.

ATTENTION! It seems in that in some processed datasets of WM1 there is no offset

## **Change in Field settings + Testing**

### 10/08/2020

change of field settings, because the M507 was not oriented to take the majority of the wind

Current installation:
- M507 points to ~250° N
- it is rotated to the N sign on the anemometer points on the bottom left when you look at the anemometer top.  
    so tentatively the rotation should be [-90, 70, 45]
    
Then done tests with an portable fan

- 13:25 horizontal wind from 230 on M507
- 13:26 vertical wind from the bottom of M507
- 13:28 horizontal wind from 340 (where the anemometer should not work properly) on M507
- 13:29 horizontal wind from N of the M506 on M506
- 13:30 horizontal wind from N of WindMaster WM1 (310° ) on WM1



Data from 13:00 till 14:00 CEST of 10 Ago 2020 **should not be considered**


### 11/08/2020

Processed data from previous night install
Added 25° offset in M506

need to confirm offset of M506

Trisonica uses the std reference system

which means that a rotation of 'z' [90] should bring back to EP reference system


### 12/08/2020

Hopefully found fan validated and tested final rotations 🎉

```python
rot_m507 = R.from_euler('XYZ', [90, 0,  135], degrees=True) 
rot_m506 = R.from_euler('z', [-90.], degrees=True)```

The Trisonica are setted on the 'std' reference system which means:

- u is positive for wind that comes from the West to the East
- v is positive for wind that comes from the South to the North

This is different from EP one but with one rotation can be converted


### 13/08/2020


trying to for 3 hours to debug an issue in wind filtering. Giving up and rewriting the preprocessing code in something easier to mantain  and debug

Basic structure of thesis and looking at fluxes

Successfully create FrankeStonica with hor wind from m6 and vertical wind from m7.. look interesting but still more work needed


### 19/08/2020

Finished refactor of processing_tools

Tested that the output of WM1 and M6 is same of old preprocessing 🎉


### 30/08/2020

during the night it rained, there may be the need to discard data for this period



### 01/09/2020

issues with the computer, it saved all data up the the 30th at midday and then only WMs. Restarted and everything works
**swapped anemometers** between 11:30 and 12:00.
TrsH had issues in restarting data till 13:00 isn't good





