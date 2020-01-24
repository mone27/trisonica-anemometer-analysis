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





