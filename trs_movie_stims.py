#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Tue Nov 23 20:44:01 2021
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Libraries for launchScan:
from builtins import str
from builtins import range
from psychopy.hardware.emulator import launchScan
# settings for launchScan:
MR_settings = {
    'TR': 2.000,     # duration (sec) per whole-brain volume
    'volumes': 5,    # number of whole-brain 3D volumes per scanning run
    'sync': 'slash', # character to use as the sync timing event; assumed to come at start of a volume
    'skip': 0,       # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
    'sound': True    # in test mode: play a tone as a reminder of scanner noise
    }
infoDlg = gui.DlgFromDict(MR_settings, title='fMRI parameters', order=['TR', 'volumes'])
if not infoDlg.OK:
    core.quit()


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'trs_movie_stims'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
concat_info = {**MR_settings, **expInfo}
# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=concat_info, runtimeInfo=None,
    originPath='/Users/ramihamati/Dropbox/Mac/Documents/PhD_Work/TRS/TRS_TRIC/movie_stims_trs/trs_movie_stims.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
trialClock = core.Clock()
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.setDefaultClock(trialClock)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=(1440, 900), fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "trial"

rest1 = visual.MovieStim3(
    win=win, name='rest1',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-1.0,
    )
clip1 = visual.MovieStim3(
    win=win, name='clip1',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip1.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-2.0,
    )
rest2 = visual.MovieStim3(
    win=win, name='rest2',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-3.0,
    )
clip2 = visual.MovieStim3(
    win=win, name='clip2',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip2.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-4.0,
    )
rest3 = visual.MovieStim3(
    win=win, name='rest3',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-5.0,
    )
clip3 = visual.MovieStim3(
    win=win, name='clip3',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip3.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-6.0,
    )
rest4 = visual.MovieStim3(
    win=win, name='rest4',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-7.0,
    )
clip4 = visual.MovieStim3(
    win=win, name='clip4',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip4.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-8.0,
    )
rest5 = visual.MovieStim3(
    win=win, name='rest5',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-9.0,
    )
clip5 = visual.MovieStim3(
    win=win, name='clip5',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip5.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-10.0,
    )
rest6 = visual.MovieStim3(
    win=win, name='rest6',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-11.0,
    )
clip6 = visual.MovieStim3(
    win=win, name='clip6',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip6.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-12.0,
    )
rest7 = visual.MovieStim3(
    win=win, name='rest7',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-13.0,
    )
clip7 = visual.MovieStim3(
    win=win, name='clip7',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/clip7.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-14.0,
    )
rest8 = visual.MovieStim3(
    win=win, name='rest8',
    noAudio = False,
    filename='/Users/ramihamati/Downloads/movies/clips_trs/rest.mp4',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=-15.0,
    )

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "trial"-------
continueRoutine = True
routineTimer.add(1723.000000)
# update component parameters for each repeat
vol = launchScan(win, MR_settings, globalClock=trialClock, esc_key='escape', mode='Scan')
# keep track of which components have finished
trialComponents = [rest1, clip1, rest2, clip2, rest3, clip3, rest4, clip4, rest5, clip5, rest6, clip6, rest7, clip7, rest8]
for thisComponent in trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "trial"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = trialClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=trialClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *rest1* updates
    if rest1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        rest1.frameNStart = frameN  # exact frame index
        rest1.tStart = t  # local t and not account for scr refresh
        rest1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest1, 'tStartRefresh')  # time at next scr refresh
        rest1.setAutoDraw(True)
    if rest1.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > rest1.tStartRefresh + 20-frameTolerance:
            # keep track of stop time/frame for later
            rest1.tStop = t  # not accounting for scr refresh
            rest1.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest1, 'tStopRefresh')  # time at next scr refresh
            rest1.setAutoDraw(False)
    
    # *clip1* updates
    if clip1.status == NOT_STARTED and tThisFlip >= 20-frameTolerance:
        # keep track of start time/frame for later
        clip1.frameNStart = frameN  # exact frame index
        clip1.tStart = t  # local t and not account for scr refresh
        clip1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip1, 'tStartRefresh')  # time at next scr refresh
        clip1.setAutoDraw(True)
    if clip1.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > clip1.tStartRefresh + 246-frameTolerance:
            # keep track of stop time/frame for later
            clip1.tStop = t  # not accounting for scr refresh
            clip1.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip1, 'tStopRefresh')  # time at next scr refresh
            clip1.setAutoDraw(False)
    
    # *rest2* updates
    if rest2.status == NOT_STARTED and tThisFlip >= 246-frameTolerance:
        # keep track of start time/frame for later
        rest2.frameNStart = frameN  # exact frame index
        rest2.tStart = t  # local t and not account for scr refresh
        rest2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest2, 'tStartRefresh')  # time at next scr refresh
        rest2.setAutoDraw(True)
    if rest2.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 266-frameTolerance:
            # keep track of stop time/frame for later
            rest2.tStop = t  # not accounting for scr refresh
            rest2.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest2, 'tStopRefresh')  # time at next scr refresh
            rest2.setAutoDraw(False)
    
    # *clip2* updates
    if clip2.status == NOT_STARTED and tThisFlip >= 266-frameTolerance:
        # keep track of start time/frame for later
        clip2.frameNStart = frameN  # exact frame index
        clip2.tStart = t  # local t and not account for scr refresh
        clip2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip2, 'tStartRefresh')  # time at next scr refresh
        clip2.setAutoDraw(True)
    if clip2.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 525-frameTolerance:
            # keep track of stop time/frame for later
            clip2.tStop = t  # not accounting for scr refresh
            clip2.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip2, 'tStopRefresh')  # time at next scr refresh
            clip2.setAutoDraw(False)
    
    # *rest3* updates
    if rest3.status == NOT_STARTED and tThisFlip >= 525-frameTolerance:
        # keep track of start time/frame for later
        rest3.frameNStart = frameN  # exact frame index
        rest3.tStart = t  # local t and not account for scr refresh
        rest3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest3, 'tStartRefresh')  # time at next scr refresh
        rest3.setAutoDraw(True)
    if rest3.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 545-frameTolerance:
            # keep track of stop time/frame for later
            rest3.tStop = t  # not accounting for scr refresh
            rest3.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest3, 'tStopRefresh')  # time at next scr refresh
            rest3.setAutoDraw(False)
    
    # *clip3* updates
    if clip3.status == NOT_STARTED and tThisFlip >= 545-frameTolerance:
        # keep track of start time/frame for later
        clip3.frameNStart = frameN  # exact frame index
        clip3.tStart = t  # local t and not account for scr refresh
        clip3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip3, 'tStartRefresh')  # time at next scr refresh
        clip3.setAutoDraw(True)
    if clip3.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 794-frameTolerance:
            # keep track of stop time/frame for later
            clip3.tStop = t  # not accounting for scr refresh
            clip3.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip3, 'tStopRefresh')  # time at next scr refresh
            clip3.setAutoDraw(False)
    
    # *rest4* updates
    if rest4.status == NOT_STARTED and tThisFlip >= 794-frameTolerance:
        # keep track of start time/frame for later
        rest4.frameNStart = frameN  # exact frame index
        rest4.tStart = t  # local t and not account for scr refresh
        rest4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest4, 'tStartRefresh')  # time at next scr refresh
        rest4.setAutoDraw(True)
    if rest4.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 814-frameTolerance:
            # keep track of stop time/frame for later
            rest4.tStop = t  # not accounting for scr refresh
            rest4.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest4, 'tStopRefresh')  # time at next scr refresh
            rest4.setAutoDraw(False)
    
    # *clip4* updates
    if clip4.status == NOT_STARTED and tThisFlip >= 814-frameTolerance:
        # keep track of start time/frame for later
        clip4.frameNStart = frameN  # exact frame index
        clip4.tStart = t  # local t and not account for scr refresh
        clip4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip4, 'tStartRefresh')  # time at next scr refresh
        clip4.setAutoDraw(True)
    if clip4.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1045-frameTolerance:
            # keep track of stop time/frame for later
            clip4.tStop = t  # not accounting for scr refresh
            clip4.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip4, 'tStopRefresh')  # time at next scr refresh
            clip4.setAutoDraw(False)
    
    # *rest5* updates
    if rest5.status == NOT_STARTED and tThisFlip >= 1045-frameTolerance:
        # keep track of start time/frame for later
        rest5.frameNStart = frameN  # exact frame index
        rest5.tStart = t  # local t and not account for scr refresh
        rest5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest5, 'tStartRefresh')  # time at next scr refresh
        rest5.setAutoDraw(True)
    if rest5.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1065-frameTolerance:
            # keep track of stop time/frame for later
            rest5.tStop = t  # not accounting for scr refresh
            rest5.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest5, 'tStopRefresh')  # time at next scr refresh
            rest5.setAutoDraw(False)
    
    # *clip5* updates
    if clip5.status == NOT_STARTED and tThisFlip >= 1065-frameTolerance:
        # keep track of start time/frame for later
        clip5.frameNStart = frameN  # exact frame index
        clip5.tStart = t  # local t and not account for scr refresh
        clip5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip5, 'tStartRefresh')  # time at next scr refresh
        clip5.setAutoDraw(True)
    if clip5.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1313-frameTolerance:
            # keep track of stop time/frame for later
            clip5.tStop = t  # not accounting for scr refresh
            clip5.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip5, 'tStopRefresh')  # time at next scr refresh
            clip5.setAutoDraw(False)
    
    # *rest6* updates
    if rest6.status == NOT_STARTED and tThisFlip >= 1313-frameTolerance:
        # keep track of start time/frame for later
        rest6.frameNStart = frameN  # exact frame index
        rest6.tStart = t  # local t and not account for scr refresh
        rest6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest6, 'tStartRefresh')  # time at next scr refresh
        rest6.setAutoDraw(True)
    if rest6.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1326-frameTolerance:
            # keep track of stop time/frame for later
            rest6.tStop = t  # not accounting for scr refresh
            rest6.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest6, 'tStopRefresh')  # time at next scr refresh
            rest6.setAutoDraw(False)
    
    # *clip6* updates
    if clip6.status == NOT_STARTED and tThisFlip >= 1326-frameTolerance:
        # keep track of start time/frame for later
        clip6.frameNStart = frameN  # exact frame index
        clip6.tStart = t  # local t and not account for scr refresh
        clip6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip6, 'tStartRefresh')  # time at next scr refresh
        clip6.setAutoDraw(True)
    if clip6.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1487-frameTolerance:
            # keep track of stop time/frame for later
            clip6.tStop = t  # not accounting for scr refresh
            clip6.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip6, 'tStopRefresh')  # time at next scr refresh
            clip6.setAutoDraw(False)
    
    # *rest7* updates
    if rest7.status == NOT_STARTED and tThisFlip >= 1487-frameTolerance:
        # keep track of start time/frame for later
        rest7.frameNStart = frameN  # exact frame index
        rest7.tStart = t  # local t and not account for scr refresh
        rest7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest7, 'tStartRefresh')  # time at next scr refresh
        rest7.setAutoDraw(True)
    if rest7.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1507-frameTolerance:
            # keep track of stop time/frame for later
            rest7.tStop = t  # not accounting for scr refresh
            rest7.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest7, 'tStopRefresh')  # time at next scr refresh
            rest7.setAutoDraw(False)
    
    # *clip7* updates
    if clip7.status == NOT_STARTED and tThisFlip >= 1507-frameTolerance:
        # keep track of start time/frame for later
        clip7.frameNStart = frameN  # exact frame index
        clip7.tStart = t  # local t and not account for scr refresh
        clip7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(clip7, 'tStartRefresh')  # time at next scr refresh
        clip7.setAutoDraw(True)
    if clip7.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1703-frameTolerance:
            # keep track of stop time/frame for later
            clip7.tStop = t  # not accounting for scr refresh
            clip7.frameNStop = frameN  # exact frame index
            win.timeOnFlip(clip7, 'tStopRefresh')  # time at next scr refresh
            clip7.setAutoDraw(False)
    
    # *rest8* updates
    if rest8.status == NOT_STARTED and tThisFlip >= 1703-frameTolerance:
        # keep track of start time/frame for later
        rest8.frameNStart = frameN  # exact frame index
        rest8.tStart = t  # local t and not account for scr refresh
        rest8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(rest8, 'tStartRefresh')  # time at next scr refresh
        rest8.setAutoDraw(True)
    if rest8.status == STARTED:
        # is it time to stop? (based on local clock)
        if tThisFlip > 1723-frameTolerance:
            # keep track of stop time/frame for later
            rest8.tStop = t  # not accounting for scr refresh
            rest8.frameNStop = frameN  # exact frame index
            win.timeOnFlip(rest8, 'tStopRefresh')  # time at next scr refresh
            rest8.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "trial"-------
for thisComponent in trialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
rest1.stop()
clip1.stop()
rest2.stop()
clip2.stop()
rest3.stop()
clip3.stop()
rest4.stop()
clip4.stop()
rest5.stop()
clip5.stop()
rest6.stop()
clip6.stop()
rest7.stop()
clip7.stop()
rest8.stop()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
