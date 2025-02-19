import FWCore.ParameterSet.Config as cms

from DQM.GEM.gemEfficiencyHarvester_cfi import gemEfficiencyHarvester
from DQM.GEM.gemEfficiencyAnalyzer_cff import gemEfficiencyAnalyzerTightGlb as _gemEfficiencyAnalyzerTightGlb
from DQM.GEM.gemEfficiencyAnalyzer_cff import gemEfficiencyAnalyzerSta as _gemEfficiencyAnalyzerSta

gemEfficiencyHarvesterTightGlb = gemEfficiencyHarvester.clone(
    folder = _gemEfficiencyAnalyzerTightGlb.folder.value()
)

gemEfficiencyHarvesterSta = gemEfficiencyHarvester.clone(
    folder = _gemEfficiencyAnalyzerSta.folder.value()
)
