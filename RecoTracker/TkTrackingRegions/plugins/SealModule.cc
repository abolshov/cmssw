#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegionProducerFromBeamSpot.h"
#include "RecoTracker/TkTrackingRegions/plugins/PointSeededTrackingRegionsProducer.h"
#include "RecoTracker/TkTrackingRegions/plugins/CandidateSeededTrackingRegionsProducer.h"
#include "RecoTracker/TkTrackingRegions/plugins/CandidatePointSeededTrackingRegionsProducer.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "GlobalTrackingRegionWithVerticesProducer.h"
#include "GlobalTrackingRegionProducer.h"
#include "AreaSeededTrackingRegionsProducer.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, GlobalTrackingRegionProducer, "GlobalRegionProducer");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  GlobalTrackingRegionProducerFromBeamSpot,
                  "GlobalRegionProducerFromBeamSpot");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  GlobalTrackingRegionWithVerticesProducer,
                  "GlobalTrackingRegionWithVerticesProducer");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  PointSeededTrackingRegionsProducer,
                  "PointSeededTrackingRegionsProducer");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  CandidateSeededTrackingRegionsProducer,
                  "CandidateSeededTrackingRegionsProducer");
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  CandidatePointSeededTrackingRegionsProducer,
                  "CandidatePointSeededTrackingRegionsProducer");

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionEDProducerT.h"
using GlobalTrackingRegionEDProducer = TrackingRegionEDProducerT<GlobalTrackingRegionProducer>;
DEFINE_FWK_MODULE(GlobalTrackingRegionEDProducer);

using GlobalTrackingRegionFromBeamSpotEDProducer = TrackingRegionEDProducerT<GlobalTrackingRegionProducerFromBeamSpot>;
DEFINE_FWK_MODULE(GlobalTrackingRegionFromBeamSpotEDProducer);

using GlobalTrackingRegionWithVerticesEDProducer = TrackingRegionEDProducerT<GlobalTrackingRegionWithVerticesProducer>;
DEFINE_FWK_MODULE(GlobalTrackingRegionWithVerticesEDProducer);

using PointSeededTrackingRegionsEDProducer = TrackingRegionEDProducerT<PointSeededTrackingRegionsProducer>;
DEFINE_FWK_MODULE(PointSeededTrackingRegionsEDProducer);

using CandidateSeededTrackingRegionsEDProducer = TrackingRegionEDProducerT<CandidateSeededTrackingRegionsProducer>;
DEFINE_FWK_MODULE(CandidateSeededTrackingRegionsEDProducer);

using CandidatePointSeededTrackingRegionsEDProducer =
    TrackingRegionEDProducerT<CandidatePointSeededTrackingRegionsProducer>;
DEFINE_FWK_MODULE(CandidatePointSeededTrackingRegionsEDProducer);

using AreaSeededTrackingRegionsEDProducer = TrackingRegionEDProducerT<AreaSeededTrackingRegionsProducer>;
DEFINE_FWK_MODULE(AreaSeededTrackingRegionsEDProducer);
