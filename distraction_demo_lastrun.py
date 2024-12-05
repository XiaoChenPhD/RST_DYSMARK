#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on ?? 31, 2024, at 10:38
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'distraction_demo'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '999',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Codes\\RST_DYSMARK\\distraction_demo_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='dkl',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'dkl'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_welcome') is None:
        # initialise key_resp_welcome
        key_resp_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_welcome',
        )
    if deviceManager.getDevice('key_resp_emotion') is None:
        # initialise key_resp_emotion
        key_resp_emotion = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_emotion',
        )
    if deviceManager.getDevice('key_resp_thinking_content') is None:
        # initialise key_resp_thinking_content
        key_resp_thinking_content = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_thinking_content',
        )
    if deviceManager.getDevice('key_resp_goodbye') is None:
        # initialise key_resp_goodbye
        key_resp_goodbye = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_goodbye',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome_screen" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='During the MRI, please focus on the idea expressed by the phrase on the screen. Please try to keep thinking about these ideas until the statement is replaced or the session has ended\n\n\nPlease try your best not to move your head or body\n\n\nPress a button to continue\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_welcome = keyboard.Keyboard(deviceName='key_resp_welcome')
    
    # --- Initialize components for Routine "refresh" ---
    text_refresh = visual.TextStim(win=win, name='text_refresh',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "wait4trigger_scanner" ---
    wait4trigger_text = visual.TextStim(win=win, name='wait4trigger_text',
        text='Please wait for the scan to start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fake_stimulus" ---
    fake_stimulus_text = visual.TextStim(win=win, name='fake_stimulus_text',
        text='Please wait for the scan to start.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "distraction" ---
    prompt_constant = visual.TextStim(win=win, name='prompt_constant',
        text='Think about:',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prompts = visual.TextStim(win=win, name='prompts',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "refresh" ---
    text_refresh = visual.TextStim(win=win, name='text_refresh',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "emotional_level" ---
    rate_emotion_prompt = visual.TextStim(win=win, name='rate_emotion_prompt',
        text='Please use a number to indicate your current feeling:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_emotion = visual.Slider(win=win, name='slider_emotion',
        startValue=5, size=(1.0, 0.1), pos=(0, -0.1), units=win.units,
        labels=["very unhappy", "", "", "", "neutral", "", "", "", "very happy"], ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.035,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    key_resp_emotion = keyboard.Keyboard(deviceName='key_resp_emotion')
    
    # --- Initialize components for Routine "refresh" ---
    text_refresh = visual.TextStim(win=win, name='text_refresh',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "think_content" ---
    main_body_questionnaire = visual.TextStim(win=win, name='main_body_questionnaire',
        text='During the MRI my thoughts:',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    item_text = visual.TextStim(win=win, name='item_text',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_thinking_content = visual.Slider(win=win, name='slider_thinking_content',
        startValue=5, size=(1.0, 0.1), pos=(0, -0.1), units=win.units,
        labels=["never", "", "", "", "sometimes", "", "", "", "always"], ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.035,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    key_resp_thinking_content = keyboard.Keyboard(deviceName='key_resp_thinking_content')
    
    # --- Initialize components for Routine "refresh" ---
    text_refresh = visual.TextStim(win=win, name='text_refresh',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "goodbye" ---
    goodbye_text = visual.TextStim(win=win, name='goodbye_text',
        text="Thank you. The current scan is over; please wait for the researcher's instructions.\n\n\n\nPress a button to continue.\n\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_goodbye = keyboard.Keyboard(deviceName='key_resp_goodbye')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome_screen" ---
    # create an object to store info about Routine welcome_screen
    welcome_screen = data.Routine(
        name='welcome_screen',
        components=[text_welcome, key_resp_welcome],
    )
    welcome_screen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_welcome
    key_resp_welcome.keys = []
    key_resp_welcome.rt = []
    _key_resp_welcome_allKeys = []
    # store start times for welcome_screen
    welcome_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome_screen.tStart = globalClock.getTime(format='float')
    welcome_screen.status = STARTED
    thisExp.addData('welcome_screen.started', welcome_screen.tStart)
    welcome_screen.maxDuration = None
    # keep track of which components have finished
    welcome_screenComponents = welcome_screen.components
    for thisComponent in welcome_screen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome_screen" ---
    welcome_screen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_welcome.started')
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_resp_welcome* updates
        waitOnFlip = False
        
        # if key_resp_welcome is starting this frame...
        if key_resp_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_welcome.frameNStart = frameN  # exact frame index
            key_resp_welcome.tStart = t  # local t and not account for scr refresh
            key_resp_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_welcome.started')
            # update status
            key_resp_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_welcome.getKeys(keyList=['a','s'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_welcome_allKeys.extend(theseKeys)
            if len(_key_resp_welcome_allKeys):
                key_resp_welcome.keys = _key_resp_welcome_allKeys[-1].name  # just the last key pressed
                key_resp_welcome.rt = _key_resp_welcome_allKeys[-1].rt
                key_resp_welcome.duration = _key_resp_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome_screen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome_screen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome_screen" ---
    for thisComponent in welcome_screen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome_screen
    welcome_screen.tStop = globalClock.getTime(format='float')
    welcome_screen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome_screen.stopped', welcome_screen.tStop)
    # check responses
    if key_resp_welcome.keys in ['', [], None]:  # No response was made
        key_resp_welcome.keys = None
    thisExp.addData('key_resp_welcome.keys',key_resp_welcome.keys)
    if key_resp_welcome.keys != None:  # we had a response
        thisExp.addData('key_resp_welcome.rt', key_resp_welcome.rt)
        thisExp.addData('key_resp_welcome.duration', key_resp_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "refresh" ---
    # create an object to store info about Routine refresh
    refresh = data.Routine(
        name='refresh',
        components=[text_refresh],
    )
    refresh.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for refresh
    refresh.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    refresh.tStart = globalClock.getTime(format='float')
    refresh.status = STARTED
    thisExp.addData('refresh.started', refresh.tStart)
    refresh.maxDuration = None
    # keep track of which components have finished
    refreshComponents = refresh.components
    for thisComponent in refresh.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "refresh" ---
    refresh.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.3:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_refresh* updates
        
        # if text_refresh is starting this frame...
        if text_refresh.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_refresh.frameNStart = frameN  # exact frame index
            text_refresh.tStart = t  # local t and not account for scr refresh
            text_refresh.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_refresh, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_refresh.started')
            # update status
            text_refresh.status = STARTED
            text_refresh.setAutoDraw(True)
        
        # if text_refresh is active this frame...
        if text_refresh.status == STARTED:
            # update params
            pass
        
        # if text_refresh is stopping this frame...
        if text_refresh.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_refresh.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                text_refresh.tStop = t  # not accounting for scr refresh
                text_refresh.tStopRefresh = tThisFlipGlobal  # on global time
                text_refresh.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_refresh.stopped')
                # update status
                text_refresh.status = FINISHED
                text_refresh.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            refresh.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in refresh.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "refresh" ---
    for thisComponent in refresh.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for refresh
    refresh.tStop = globalClock.getTime(format='float')
    refresh.tStopRefresh = tThisFlipGlobal
    thisExp.addData('refresh.stopped', refresh.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if refresh.maxDurationReached:
        routineTimer.addTime(-refresh.maxDuration)
    elif refresh.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.300000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "wait4trigger_scanner" ---
    # create an object to store info about Routine wait4trigger_scanner
    wait4trigger_scanner = data.Routine(
        name='wait4trigger_scanner',
        components=[wait4trigger_text],
    )
    wait4trigger_scanner.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for wait4trigger_scanner
    wait4trigger_scanner.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    wait4trigger_scanner.tStart = globalClock.getTime(format='float')
    wait4trigger_scanner.status = STARTED
    thisExp.addData('wait4trigger_scanner.started', wait4trigger_scanner.tStart)
    wait4trigger_scanner.maxDuration = None
    # keep track of which components have finished
    wait4trigger_scannerComponents = wait4trigger_scanner.components
    for thisComponent in wait4trigger_scanner.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "wait4trigger_scanner" ---
    wait4trigger_scanner.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *wait4trigger_text* updates
        
        # if wait4trigger_text is starting this frame...
        if wait4trigger_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            wait4trigger_text.frameNStart = frameN  # exact frame index
            wait4trigger_text.tStart = t  # local t and not account for scr refresh
            wait4trigger_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wait4trigger_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wait4trigger_text.started')
            # update status
            wait4trigger_text.status = STARTED
            wait4trigger_text.setAutoDraw(True)
        
        # if wait4trigger_text is active this frame...
        if wait4trigger_text.status == STARTED:
            # update params
            pass
        
        # if wait4trigger_text is stopping this frame...
        if wait4trigger_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > wait4trigger_text.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                wait4trigger_text.tStop = t  # not accounting for scr refresh
                wait4trigger_text.tStopRefresh = tThisFlipGlobal  # on global time
                wait4trigger_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wait4trigger_text.stopped')
                # update status
                wait4trigger_text.status = FINISHED
                wait4trigger_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            wait4trigger_scanner.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in wait4trigger_scanner.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "wait4trigger_scanner" ---
    for thisComponent in wait4trigger_scanner.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for wait4trigger_scanner
    wait4trigger_scanner.tStop = globalClock.getTime(format='float')
    wait4trigger_scanner.tStopRefresh = tThisFlipGlobal
    thisExp.addData('wait4trigger_scanner.stopped', wait4trigger_scanner.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if wait4trigger_scanner.maxDurationReached:
        routineTimer.addTime(-wait4trigger_scanner.maxDuration)
    elif wait4trigger_scanner.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "fake_stimulus" ---
    # create an object to store info about Routine fake_stimulus
    fake_stimulus = data.Routine(
        name='fake_stimulus',
        components=[fake_stimulus_text],
    )
    fake_stimulus.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for fake_stimulus
    fake_stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    fake_stimulus.tStart = globalClock.getTime(format='float')
    fake_stimulus.status = STARTED
    thisExp.addData('fake_stimulus.started', fake_stimulus.tStart)
    fake_stimulus.maxDuration = None
    # keep track of which components have finished
    fake_stimulusComponents = fake_stimulus.components
    for thisComponent in fake_stimulus.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "fake_stimulus" ---
    fake_stimulus.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fake_stimulus_text* updates
        
        # if fake_stimulus_text is starting this frame...
        if fake_stimulus_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fake_stimulus_text.frameNStart = frameN  # exact frame index
            fake_stimulus_text.tStart = t  # local t and not account for scr refresh
            fake_stimulus_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fake_stimulus_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fake_stimulus_text.started')
            # update status
            fake_stimulus_text.status = STARTED
            fake_stimulus_text.setAutoDraw(True)
        
        # if fake_stimulus_text is active this frame...
        if fake_stimulus_text.status == STARTED:
            # update params
            pass
        
        # if fake_stimulus_text is stopping this frame...
        if fake_stimulus_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fake_stimulus_text.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                fake_stimulus_text.tStop = t  # not accounting for scr refresh
                fake_stimulus_text.tStopRefresh = tThisFlipGlobal  # on global time
                fake_stimulus_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fake_stimulus_text.stopped')
                # update status
                fake_stimulus_text.status = FINISHED
                fake_stimulus_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            fake_stimulus.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in fake_stimulus.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "fake_stimulus" ---
    for thisComponent in fake_stimulus.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for fake_stimulus
    fake_stimulus.tStop = globalClock.getTime(format='float')
    fake_stimulus.tStopRefresh = tThisFlipGlobal
    thisExp.addData('fake_stimulus.stopped', fake_stimulus.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if fake_stimulus.maxDurationReached:
        routineTimer.addTime(-fake_stimulus.maxDuration)
    elif fake_stimulus.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    distraction_loop = data.TrialHandler2(
        name='distraction_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('distraction_prompts.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(distraction_loop)  # add the loop to the experiment
    thisDistraction_loop = distraction_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDistraction_loop.rgb)
    if thisDistraction_loop != None:
        for paramName in thisDistraction_loop:
            globals()[paramName] = thisDistraction_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisDistraction_loop in distraction_loop:
        currentLoop = distraction_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisDistraction_loop.rgb)
        if thisDistraction_loop != None:
            for paramName in thisDistraction_loop:
                globals()[paramName] = thisDistraction_loop[paramName]
        
        # --- Prepare to start Routine "distraction" ---
        # create an object to store info about Routine distraction
        distraction = data.Routine(
            name='distraction',
            components=[prompt_constant, prompts],
        )
        distraction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prompt_constant.setPos((-0.5, 0.2))
        prompts.setText(prompt)
        # Run 'Begin Routine' code from code_prompts
        prompts.alignText = 'center'
        # store start times for distraction
        distraction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        distraction.tStart = globalClock.getTime(format='float')
        distraction.status = STARTED
        thisExp.addData('distraction.started', distraction.tStart)
        distraction.maxDuration = None
        # keep track of which components have finished
        distractionComponents = distraction.components
        for thisComponent in distraction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "distraction" ---
        # if trial has changed, end Routine now
        if isinstance(distraction_loop, data.TrialHandler2) and thisDistraction_loop.thisN != distraction_loop.thisTrial.thisN:
            continueRoutine = False
        distraction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prompt_constant* updates
            
            # if prompt_constant is starting this frame...
            if prompt_constant.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompt_constant.frameNStart = frameN  # exact frame index
                prompt_constant.tStart = t  # local t and not account for scr refresh
                prompt_constant.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompt_constant, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompt_constant.started')
                # update status
                prompt_constant.status = STARTED
                prompt_constant.setAutoDraw(True)
            
            # if prompt_constant is active this frame...
            if prompt_constant.status == STARTED:
                # update params
                pass
            
            # if prompt_constant is stopping this frame...
            if prompt_constant.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prompt_constant.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    prompt_constant.tStop = t  # not accounting for scr refresh
                    prompt_constant.tStopRefresh = tThisFlipGlobal  # on global time
                    prompt_constant.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prompt_constant.stopped')
                    # update status
                    prompt_constant.status = FINISHED
                    prompt_constant.setAutoDraw(False)
            
            # *prompts* updates
            
            # if prompts is starting this frame...
            if prompts.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prompts.frameNStart = frameN  # exact frame index
                prompts.tStart = t  # local t and not account for scr refresh
                prompts.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prompts, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prompts.started')
                # update status
                prompts.status = STARTED
                prompts.setAutoDraw(True)
            
            # if prompts is active this frame...
            if prompts.status == STARTED:
                # update params
                pass
            
            # if prompts is stopping this frame...
            if prompts.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prompts.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    prompts.tStop = t  # not accounting for scr refresh
                    prompts.tStopRefresh = tThisFlipGlobal  # on global time
                    prompts.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prompts.stopped')
                    # update status
                    prompts.status = FINISHED
                    prompts.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                distraction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in distraction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "distraction" ---
        for thisComponent in distraction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for distraction
        distraction.tStop = globalClock.getTime(format='float')
        distraction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('distraction.stopped', distraction.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if distraction.maxDurationReached:
            routineTimer.addTime(-distraction.maxDuration)
        elif distraction.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'distraction_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "refresh" ---
    # create an object to store info about Routine refresh
    refresh = data.Routine(
        name='refresh',
        components=[text_refresh],
    )
    refresh.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for refresh
    refresh.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    refresh.tStart = globalClock.getTime(format='float')
    refresh.status = STARTED
    thisExp.addData('refresh.started', refresh.tStart)
    refresh.maxDuration = None
    # keep track of which components have finished
    refreshComponents = refresh.components
    for thisComponent in refresh.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "refresh" ---
    refresh.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.3:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_refresh* updates
        
        # if text_refresh is starting this frame...
        if text_refresh.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_refresh.frameNStart = frameN  # exact frame index
            text_refresh.tStart = t  # local t and not account for scr refresh
            text_refresh.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_refresh, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_refresh.started')
            # update status
            text_refresh.status = STARTED
            text_refresh.setAutoDraw(True)
        
        # if text_refresh is active this frame...
        if text_refresh.status == STARTED:
            # update params
            pass
        
        # if text_refresh is stopping this frame...
        if text_refresh.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_refresh.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                text_refresh.tStop = t  # not accounting for scr refresh
                text_refresh.tStopRefresh = tThisFlipGlobal  # on global time
                text_refresh.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_refresh.stopped')
                # update status
                text_refresh.status = FINISHED
                text_refresh.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            refresh.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in refresh.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "refresh" ---
    for thisComponent in refresh.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for refresh
    refresh.tStop = globalClock.getTime(format='float')
    refresh.tStopRefresh = tThisFlipGlobal
    thisExp.addData('refresh.stopped', refresh.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if refresh.maxDurationReached:
        routineTimer.addTime(-refresh.maxDuration)
    elif refresh.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.300000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "emotional_level" ---
    # create an object to store info about Routine emotional_level
    emotional_level = data.Routine(
        name='emotional_level',
        components=[rate_emotion_prompt, slider_emotion, key_resp_emotion],
    )
    emotional_level.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    slider_emotion.reset()
    # Run 'Begin Routine' code from code_emotion
    event.clearEvents('keyboard')
    slider_emotion.markerPos = 5
    # create starting attributes for key_resp_emotion
    key_resp_emotion.keys = []
    key_resp_emotion.rt = []
    _key_resp_emotion_allKeys = []
    # store start times for emotional_level
    emotional_level.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    emotional_level.tStart = globalClock.getTime(format='float')
    emotional_level.status = STARTED
    thisExp.addData('emotional_level.started', emotional_level.tStart)
    emotional_level.maxDuration = None
    # keep track of which components have finished
    emotional_levelComponents = emotional_level.components
    for thisComponent in emotional_level.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "emotional_level" ---
    emotional_level.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rate_emotion_prompt* updates
        
        # if rate_emotion_prompt is starting this frame...
        if rate_emotion_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rate_emotion_prompt.frameNStart = frameN  # exact frame index
            rate_emotion_prompt.tStart = t  # local t and not account for scr refresh
            rate_emotion_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rate_emotion_prompt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rate_emotion_prompt.started')
            # update status
            rate_emotion_prompt.status = STARTED
            rate_emotion_prompt.setAutoDraw(True)
        
        # if rate_emotion_prompt is active this frame...
        if rate_emotion_prompt.status == STARTED:
            # update params
            pass
        
        # *slider_emotion* updates
        
        # if slider_emotion is starting this frame...
        if slider_emotion.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_emotion.frameNStart = frameN  # exact frame index
            slider_emotion.tStart = t  # local t and not account for scr refresh
            slider_emotion.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_emotion, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_emotion.started')
            # update status
            slider_emotion.status = STARTED
            slider_emotion.setAutoDraw(True)
        
        # if slider_emotion is active this frame...
        if slider_emotion.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from code_emotion
        keys = event.getKeys()
        
        if len(keys):
            if 'k' in keys:
                slider_emotion.markerPos = slider_emotion.markerPos - 1
            elif 'l' in keys:
                slider_emotion.markerPos = slider_emotion.markerPos  + 1 
        
        # *key_resp_emotion* updates
        waitOnFlip = False
        
        # if key_resp_emotion is starting this frame...
        if key_resp_emotion.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_emotion.frameNStart = frameN  # exact frame index
            key_resp_emotion.tStart = t  # local t and not account for scr refresh
            key_resp_emotion.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_emotion, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_emotion.started')
            # update status
            key_resp_emotion.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_emotion.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_emotion.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_emotion.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_emotion.getKeys(keyList=['a','s'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_emotion_allKeys.extend(theseKeys)
            if len(_key_resp_emotion_allKeys):
                key_resp_emotion.keys = _key_resp_emotion_allKeys[-1].name  # just the last key pressed
                key_resp_emotion.rt = _key_resp_emotion_allKeys[-1].rt
                key_resp_emotion.duration = _key_resp_emotion_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            emotional_level.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in emotional_level.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "emotional_level" ---
    for thisComponent in emotional_level.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for emotional_level
    emotional_level.tStop = globalClock.getTime(format='float')
    emotional_level.tStopRefresh = tThisFlipGlobal
    thisExp.addData('emotional_level.stopped', emotional_level.tStop)
    thisExp.addData('slider_emotion.response', slider_emotion.getRating())
    thisExp.addData('slider_emotion.rt', slider_emotion.getRT())
    # Run 'End Routine' code from code_emotion
    thisExp.addData("Rating", slider_emotion.markerPos)
    # check responses
    if key_resp_emotion.keys in ['', [], None]:  # No response was made
        key_resp_emotion.keys = None
    thisExp.addData('key_resp_emotion.keys',key_resp_emotion.keys)
    if key_resp_emotion.keys != None:  # we had a response
        thisExp.addData('key_resp_emotion.rt', key_resp_emotion.rt)
        thisExp.addData('key_resp_emotion.duration', key_resp_emotion.duration)
    thisExp.nextEntry()
    # the Routine "emotional_level" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "refresh" ---
    # create an object to store info about Routine refresh
    refresh = data.Routine(
        name='refresh',
        components=[text_refresh],
    )
    refresh.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for refresh
    refresh.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    refresh.tStart = globalClock.getTime(format='float')
    refresh.status = STARTED
    thisExp.addData('refresh.started', refresh.tStart)
    refresh.maxDuration = None
    # keep track of which components have finished
    refreshComponents = refresh.components
    for thisComponent in refresh.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "refresh" ---
    refresh.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.3:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_refresh* updates
        
        # if text_refresh is starting this frame...
        if text_refresh.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_refresh.frameNStart = frameN  # exact frame index
            text_refresh.tStart = t  # local t and not account for scr refresh
            text_refresh.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_refresh, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_refresh.started')
            # update status
            text_refresh.status = STARTED
            text_refresh.setAutoDraw(True)
        
        # if text_refresh is active this frame...
        if text_refresh.status == STARTED:
            # update params
            pass
        
        # if text_refresh is stopping this frame...
        if text_refresh.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_refresh.tStartRefresh + 0.3-frameTolerance:
                # keep track of stop time/frame for later
                text_refresh.tStop = t  # not accounting for scr refresh
                text_refresh.tStopRefresh = tThisFlipGlobal  # on global time
                text_refresh.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_refresh.stopped')
                # update status
                text_refresh.status = FINISHED
                text_refresh.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            refresh.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in refresh.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "refresh" ---
    for thisComponent in refresh.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for refresh
    refresh.tStop = globalClock.getTime(format='float')
    refresh.tStopRefresh = tThisFlipGlobal
    thisExp.addData('refresh.stopped', refresh.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if refresh.maxDurationReached:
        routineTimer.addTime(-refresh.maxDuration)
    elif refresh.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.300000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    think_content_loop = data.TrialHandler2(
        name='think_content_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('think_content_condition.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(think_content_loop)  # add the loop to the experiment
    thisThink_content_loop = think_content_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisThink_content_loop.rgb)
    if thisThink_content_loop != None:
        for paramName in thisThink_content_loop:
            globals()[paramName] = thisThink_content_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisThink_content_loop in think_content_loop:
        currentLoop = think_content_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisThink_content_loop.rgb)
        if thisThink_content_loop != None:
            for paramName in thisThink_content_loop:
                globals()[paramName] = thisThink_content_loop[paramName]
        
        # --- Prepare to start Routine "think_content" ---
        # create an object to store info about Routine think_content
        think_content = data.Routine(
            name='think_content',
            components=[main_body_questionnaire, item_text, slider_thinking_content, key_resp_thinking_content],
        )
        think_content.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        item_text.setText(item)
        slider_thinking_content.reset()
        # Run 'Begin Routine' code from code_thinking_content
        event.clearEvents('keyboard')
        slider_thinking_content.markerPos = 5
        # create starting attributes for key_resp_thinking_content
        key_resp_thinking_content.keys = []
        key_resp_thinking_content.rt = []
        _key_resp_thinking_content_allKeys = []
        # store start times for think_content
        think_content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        think_content.tStart = globalClock.getTime(format='float')
        think_content.status = STARTED
        thisExp.addData('think_content.started', think_content.tStart)
        think_content.maxDuration = None
        # keep track of which components have finished
        think_contentComponents = think_content.components
        for thisComponent in think_content.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "think_content" ---
        # if trial has changed, end Routine now
        if isinstance(think_content_loop, data.TrialHandler2) and thisThink_content_loop.thisN != think_content_loop.thisTrial.thisN:
            continueRoutine = False
        think_content.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *main_body_questionnaire* updates
            
            # if main_body_questionnaire is starting this frame...
            if main_body_questionnaire.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                main_body_questionnaire.frameNStart = frameN  # exact frame index
                main_body_questionnaire.tStart = t  # local t and not account for scr refresh
                main_body_questionnaire.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(main_body_questionnaire, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'main_body_questionnaire.started')
                # update status
                main_body_questionnaire.status = STARTED
                main_body_questionnaire.setAutoDraw(True)
            
            # if main_body_questionnaire is active this frame...
            if main_body_questionnaire.status == STARTED:
                # update params
                pass
            
            # *item_text* updates
            
            # if item_text is starting this frame...
            if item_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                item_text.frameNStart = frameN  # exact frame index
                item_text.tStart = t  # local t and not account for scr refresh
                item_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item_text.started')
                # update status
                item_text.status = STARTED
                item_text.setAutoDraw(True)
            
            # if item_text is active this frame...
            if item_text.status == STARTED:
                # update params
                pass
            
            # *slider_thinking_content* updates
            
            # if slider_thinking_content is starting this frame...
            if slider_thinking_content.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_thinking_content.frameNStart = frameN  # exact frame index
                slider_thinking_content.tStart = t  # local t and not account for scr refresh
                slider_thinking_content.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_thinking_content, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_thinking_content.started')
                # update status
                slider_thinking_content.status = STARTED
                slider_thinking_content.setAutoDraw(True)
            
            # if slider_thinking_content is active this frame...
            if slider_thinking_content.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_thinking_content
            keys = event.getKeys()
            
            if len(keys):
                if 'k' in keys:
                    slider_thinking_content.markerPos = slider_thinking_content.markerPos - 1
                elif 'l' in keys:
                    slider_thinking_content.markerPos = slider_thinking_content.markerPos  + 1 
            
            # *key_resp_thinking_content* updates
            waitOnFlip = False
            
            # if key_resp_thinking_content is starting this frame...
            if key_resp_thinking_content.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_thinking_content.frameNStart = frameN  # exact frame index
                key_resp_thinking_content.tStart = t  # local t and not account for scr refresh
                key_resp_thinking_content.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_thinking_content, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_thinking_content.started')
                # update status
                key_resp_thinking_content.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_thinking_content.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_thinking_content.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_thinking_content.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_thinking_content.getKeys(keyList=['a','s'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_thinking_content_allKeys.extend(theseKeys)
                if len(_key_resp_thinking_content_allKeys):
                    key_resp_thinking_content.keys = _key_resp_thinking_content_allKeys[-1].name  # just the last key pressed
                    key_resp_thinking_content.rt = _key_resp_thinking_content_allKeys[-1].rt
                    key_resp_thinking_content.duration = _key_resp_thinking_content_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                think_content.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in think_content.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "think_content" ---
        for thisComponent in think_content.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for think_content
        think_content.tStop = globalClock.getTime(format='float')
        think_content.tStopRefresh = tThisFlipGlobal
        thisExp.addData('think_content.stopped', think_content.tStop)
        think_content_loop.addData('slider_thinking_content.response', slider_thinking_content.getRating())
        think_content_loop.addData('slider_thinking_content.rt', slider_thinking_content.getRT())
        # Run 'End Routine' code from code_thinking_content
        thisExp.addData("Rating", slider_thinking_content.markerPos)
        # check responses
        if key_resp_thinking_content.keys in ['', [], None]:  # No response was made
            key_resp_thinking_content.keys = None
        think_content_loop.addData('key_resp_thinking_content.keys',key_resp_thinking_content.keys)
        if key_resp_thinking_content.keys != None:  # we had a response
            think_content_loop.addData('key_resp_thinking_content.rt', key_resp_thinking_content.rt)
            think_content_loop.addData('key_resp_thinking_content.duration', key_resp_thinking_content.duration)
        # the Routine "think_content" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "refresh" ---
        # create an object to store info about Routine refresh
        refresh = data.Routine(
            name='refresh',
            components=[text_refresh],
        )
        refresh.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for refresh
        refresh.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        refresh.tStart = globalClock.getTime(format='float')
        refresh.status = STARTED
        thisExp.addData('refresh.started', refresh.tStart)
        refresh.maxDuration = None
        # keep track of which components have finished
        refreshComponents = refresh.components
        for thisComponent in refresh.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "refresh" ---
        # if trial has changed, end Routine now
        if isinstance(think_content_loop, data.TrialHandler2) and thisThink_content_loop.thisN != think_content_loop.thisTrial.thisN:
            continueRoutine = False
        refresh.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.3:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_refresh* updates
            
            # if text_refresh is starting this frame...
            if text_refresh.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_refresh.frameNStart = frameN  # exact frame index
                text_refresh.tStart = t  # local t and not account for scr refresh
                text_refresh.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_refresh, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_refresh.started')
                # update status
                text_refresh.status = STARTED
                text_refresh.setAutoDraw(True)
            
            # if text_refresh is active this frame...
            if text_refresh.status == STARTED:
                # update params
                pass
            
            # if text_refresh is stopping this frame...
            if text_refresh.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_refresh.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_refresh.tStop = t  # not accounting for scr refresh
                    text_refresh.tStopRefresh = tThisFlipGlobal  # on global time
                    text_refresh.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_refresh.stopped')
                    # update status
                    text_refresh.status = FINISHED
                    text_refresh.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                refresh.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in refresh.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "refresh" ---
        for thisComponent in refresh.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for refresh
        refresh.tStop = globalClock.getTime(format='float')
        refresh.tStopRefresh = tThisFlipGlobal
        thisExp.addData('refresh.stopped', refresh.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if refresh.maxDurationReached:
            routineTimer.addTime(-refresh.maxDuration)
        elif refresh.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.300000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'think_content_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "goodbye" ---
    # create an object to store info about Routine goodbye
    goodbye = data.Routine(
        name='goodbye',
        components=[goodbye_text, key_resp_goodbye],
    )
    goodbye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_goodbye
    key_resp_goodbye.keys = []
    key_resp_goodbye.rt = []
    _key_resp_goodbye_allKeys = []
    # store start times for goodbye
    goodbye.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    goodbye.tStart = globalClock.getTime(format='float')
    goodbye.status = STARTED
    thisExp.addData('goodbye.started', goodbye.tStart)
    goodbye.maxDuration = None
    # keep track of which components have finished
    goodbyeComponents = goodbye.components
    for thisComponent in goodbye.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "goodbye" ---
    goodbye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *goodbye_text* updates
        
        # if goodbye_text is starting this frame...
        if goodbye_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            goodbye_text.frameNStart = frameN  # exact frame index
            goodbye_text.tStart = t  # local t and not account for scr refresh
            goodbye_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(goodbye_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'goodbye_text.started')
            # update status
            goodbye_text.status = STARTED
            goodbye_text.setAutoDraw(True)
        
        # if goodbye_text is active this frame...
        if goodbye_text.status == STARTED:
            # update params
            pass
        
        # if goodbye_text is stopping this frame...
        if goodbye_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > goodbye_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                goodbye_text.tStop = t  # not accounting for scr refresh
                goodbye_text.tStopRefresh = tThisFlipGlobal  # on global time
                goodbye_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'goodbye_text.stopped')
                # update status
                goodbye_text.status = FINISHED
                goodbye_text.setAutoDraw(False)
        
        # *key_resp_goodbye* updates
        waitOnFlip = False
        
        # if key_resp_goodbye is starting this frame...
        if key_resp_goodbye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_goodbye.frameNStart = frameN  # exact frame index
            key_resp_goodbye.tStart = t  # local t and not account for scr refresh
            key_resp_goodbye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_goodbye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_goodbye.started')
            # update status
            key_resp_goodbye.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_goodbye.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_goodbye.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_goodbye.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_goodbye.getKeys(keyList=['a','s'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_goodbye_allKeys.extend(theseKeys)
            if len(_key_resp_goodbye_allKeys):
                key_resp_goodbye.keys = _key_resp_goodbye_allKeys[-1].name  # just the last key pressed
                key_resp_goodbye.rt = _key_resp_goodbye_allKeys[-1].rt
                key_resp_goodbye.duration = _key_resp_goodbye_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            goodbye.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in goodbye.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "goodbye" ---
    for thisComponent in goodbye.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for goodbye
    goodbye.tStop = globalClock.getTime(format='float')
    goodbye.tStopRefresh = tThisFlipGlobal
    thisExp.addData('goodbye.stopped', goodbye.tStop)
    # check responses
    if key_resp_goodbye.keys in ['', [], None]:  # No response was made
        key_resp_goodbye.keys = None
    thisExp.addData('key_resp_goodbye.keys',key_resp_goodbye.keys)
    if key_resp_goodbye.keys != None:  # we had a response
        thisExp.addData('key_resp_goodbye.rt', key_resp_goodbye.rt)
        thisExp.addData('key_resp_goodbye.duration', key_resp_goodbye.duration)
    thisExp.nextEntry()
    # the Routine "goodbye" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
