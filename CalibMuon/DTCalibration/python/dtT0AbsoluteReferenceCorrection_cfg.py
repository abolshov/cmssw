import FWCore.ParameterSet.Config as cms

process = cms.Process("DTT0Correction")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtT0AbsoluteReferenceCorrection')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.DTGeometryESModule.fromDDD = False

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),

    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0')
    ))
)
process.PoolDBOutputService.connect = cms.string('sqlite_file:t0.db')

process.load("CalibMuon.DTCalibration.dtT0AbsoluteReferenceCorrection_cfi")
process.dtT0AbsoluteReferenceCorrection.correctionAlgoConfig.calibChamber = 'All'
process.dtT0AbsoluteReferenceCorrection.correctionAlgoConfig.reference = 640.

process.p = cms.Path(process.dtT0AbsoluteReferenceCorrection)
