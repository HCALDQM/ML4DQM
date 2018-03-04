#-------------------------------------
#Hcal DQM Application using New DQM Sources/Clients
#Online Mode
#-------------------------------------

#-------------------------------------
#Standard Python Imports
#-------------------------------------
import os, sys, socket, string

#-------------------------------------
#Standard CMSSW Imports/Definitions
#-------------------------------------
import FWCore.ParameterSet.Config as cms

#
# these Modifiers are like eras as well, for more info check
# Configuration/StandardSequences/python/Eras.py
# PRocess accepts a (*list) of modifiers
#
from Configuration.StandardSequences.Eras import eras
process      = cms.Process('HCALDQM', eras.Run2_2017)
subsystem    = 'HcalAll'
cmssw        = os.getenv("CMSSW_VERSION").split("_")
debugstr     = "### HcalDQM::cfg::DEBUG: "
warnstr      = "### HcalDQM::cfg::WARN: "
errorstr     = "### HcalDQM::cfg::ERROR:"
useOfflineGT = True
useFileInput = True

#-------------------------------------
#Local Source definition
#-------------------------------------
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),
                            lumisToProcess = cms.untracked.VLuminosityBlockRange()
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

#-------------------------------------
#Central DQM Stuff imports
#-------------------------------------
from DQMServices.Core.DQMStore_cfi import *

if useOfflineGT:
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
    process.GlobalTag.globaltag = '100X_dataRun2_HLT_v1'
else:
    process.load('DQM.Integration.config.FrontierCondition_GT_cfi')

if useFileInput:
    #process.load("DQM.Integration.config.fileinputsource_cfi")
    #process.source.fileNames = cms.untracked.vstring('file:/eos/cms/store/group/dpg_hcal/comm_hcal/ML4DQM/rawData/Ephemeral_PU56-58_305636000.root')
    process.source.fileNames = cms.untracked.vstring(INPUTLIST)
    process.source.fileNames.extend(SECONDARY)
    process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange(LUMIRANGE)
    
else:
    process.load('DQM.Integration.config.inputsource_cfi')

process.load('DQMServices.Components.DQMFileSaver_cfi')
#-------------------------------------
#Central DQM Customization
#-------------------------------------
process.DQMStore = cms.Service("DQMStore")
process.dqmSaver.workflow = "/HcalAll/Run2017/Offline"
#process.dqmSaver.dirName = "outFiles"
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)

#-------------------------------------
#CMSSW/Hcal non-DQM Related Module import
#-------------------------------------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
process.load('CondCore.CondDB.CondDB_cfi')

#-------------------------------------
#CMSSW/Hcal non-DQM Related Module Settings
#-> runType
#-> Generic Input tag for the Raw Collection
#-> cmssw version
#-> Turn off default blocking of dead channels from rechit collection
#-> Drop Channel Status Bits (had benn 'HcalCellOff', "HcalCellDead")
#-> For Trigger Primitives Emulation
#-> L1 GT setting
#-> Rename the hbheprereco to hbhereco
#-------------------------------------
cmssw= os.getenv("CMSSW_VERSION").split("_")
rawTag= cms.InputTag("rawDataCollector")
rawTagUntracked= cms.untracked.InputTag("rawDataCollector")

#set the tag for default unpacker
process.hcalDigis.InputLabel = rawTag

#-------------------------------------
#Hcal DQM Tasks and Clients import
#New Style
#-------------------------------------
process.load("DQM.HcalTasks.DigiTask")
process.load('DQM.HcalTasks.RecHitTask')

#-------------------------------------
#Settings for the Primary Modules
#-------------------------------------
process.digiTask.subsystem = cms.untracked.string(subsystem)

process.recHitTask.tagHBHE = cms.untracked.InputTag("hbheprereco")
process.recHitTask.tagHO = cms.untracked.InputTag("horeco")
process.recHitTask.tagHF = cms.untracked.InputTag("hfreco")
process.recHitTask.tagRaw = rawTagUntracked
process.recHitTask.subsystem = cms.untracked.string(subsystem)

#-------------------------------------
#Hcal DQM Tasks/Clients Sequences Definition
#-------------------------------------
process.tasksPath = cms.Path(
    process.digiTask
    *process.recHitTask
)


#-------------------------------------
process.digiPath = cms.Path(
    process.hcalDigis
)

process.recoPath = cms.Path(
    process.hcalLocalRecoSequence
)

process.dqmPath = cms.EndPath(
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.digiPath,
    process.recoPath,
    process.tasksPath,
    process.dqmPath
)

#-------------------------------------
#Scheduling and Process Customizations
#-------------------------------------
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring(
        "ProductNotFound",
        "TooManyProducts",
        "TooFewProducts"
        )
)
process.options.wantSummary = cms.untracked.bool(True)
