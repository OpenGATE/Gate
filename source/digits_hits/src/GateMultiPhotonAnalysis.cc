/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateMultiPhotonAnalysis.hh"

#include "GateActions.hh"
#include "GateAnalysis.hh"
#include "GateDigitizerMgr.hh"
#include "GateHit.hh"
#include "GateMultiPhotonAnalysisMessenger.hh"
#include "GateMultiPhotonTrajectoryNavigator.hh"
#include "GateOutputMgr.hh"
#include "GatePhantomHit.hh"
#include "GateRunManager.hh"
#include "GateSourceMgr.hh"

#include "G4Event.hh"
#include "G4HCofThisEvent.hh"
#include "G4Navigator.hh"
#include "G4Run.hh"
#include "G4TrajectoryContainer.hh"
#include "G4TransportationManager.hh"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct TimelineEntry {
  double time = 0.0;
  std::size_t sequence = 0;
  bool is_phantom = false;
  int track_id = 0;
  GatePhantomHit *phantom_hit = 0;
  GateHit *crystal_hit = 0;
};

struct EventContext {
  G4int event_id = 0;
  G4int run_id = 0;
  G4int source_id = -1;
  G4ThreeVector source_vertex = G4ThreeVector(-1, -1, -1);
};

bool EventHasProcessableHits(const std::vector<GateHitsCollection *> &CHC_vector,
                             GatePhantomHitsCollection *PHC) {
  if (PHC && PHC->entries() > 0) {
    return true;
  }

  for (std::size_t i = 0; i < CHC_vector.size(); ++i) {
    GateHitsCollection *CHC = CHC_vector[i];
    if (CHC && CHC->entries() > 0) {
      return true;
    }
  }

  return false;
}

std::vector<TimelineEntry> BuildBasePhantomTimeline(GatePhantomHitsCollection *PHC) {
  std::vector<TimelineEntry> basePhantomTimeline;
  if (!PHC) {
    return basePhantomTimeline;
  }

  std::size_t sequence = 0;
  const G4int NpHits = PHC->entries();
  basePhantomTimeline.reserve(static_cast<std::size_t>(NpHits));

  for (G4int iPHit = 0; iPHit < NpHits; ++iPHit) {
    GatePhantomHit *phantomHit = (*PHC)[iPHit];
    if (!phantomHit) {
      continue;
    }

    TimelineEntry entry;
    entry.time = phantomHit->GetTime();
    entry.sequence = sequence++;
    entry.is_phantom = true;
    entry.track_id = phantomHit->GetTrackID();
    entry.phantom_hit = phantomHit;
    basePhantomTimeline.push_back(entry);
  }

  return basePhantomTimeline;
}

std::string LocateVolumeName(const G4ThreeVector &position) {
  G4Navigator *navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
  G4ThreeVector direction(0., 0., 0.);
  G4VPhysicalVolume *volume = navigator->LocateGlobalPointAndSetup(position, &direction, false);
  if (!volume) {
    return MultiPhotonAnalysisHelpers::kNoVolumeName;
  }
  return volume->GetName();
}

int CountSeptalHits(GatePhantomHitsCollection *PHC) {
  // Septal penetration is configured on the GateAnalysis module (setSeptalVolumeName and
  // recordSeptalPenetration), so the settings are read from there - the same way GateRootDefs
  // does it when deciding whether to create the septalNb branch.
  // The counting rule is copied from GateAnalysis: every phantom hit recorded in the septal
  // volume increments the counter, with no per-photon attribution, and the resulting event
  // total is written to all crystal hits.
  GateAnalysis *analysis =
      dynamic_cast<GateAnalysis *>(GateOutputMgr::GetInstance()->GetModule("analysis"));
  if (!analysis || !analysis->GetRecordSeptalFlag() || !PHC) {
    return 0;
  }

  const G4String &septalVolumeName = analysis->GetSeptalPhysVolumeName();

  int septalNb = 0;
  const G4int NpHits = PHC->entries();
  for (G4int iPHit = 0; iPHit < NpHits; ++iPHit) {
    GatePhantomHit *phantomHit = (*PHC)[iPHit];
    if (phantomHit && phantomHit->GetPhysVolName() == septalVolumeName) {
      ++septalNb;
    }
  }

  return septalNb;
}

std::unordered_map<int, MultiPhotonAnalysisHelpers::PhantomStatistics> ComputePhantomTotals(
    GatePhantomHitsCollection *PHC,
    GateMultiPhotonTrajectoryNavigator *trajectoryNavigator) {
  // Phantom counters are event totals, exactly like in GateAnalysis: they are computed in
  // a pass that precedes the crystal hits, so every crystal hit of a gamma gets the same value.
  // Only hits of the reference photon itself are counted (attribution by exact track ID).
  std::unordered_map<int, MultiPhotonAnalysisHelpers::PhantomStatistics> totals;
  if (!PHC) {
    return totals;
  }

  const G4int NpHits = PHC->entries();
  for (G4int iPHit = 0; iPHit < NpHits; ++iPHit) {
    GatePhantomHit *phantomHit = (*PHC)[iPHit];
    if (!phantomHit) {
      continue;
    }

    const G4int trackID = phantomHit->GetTrackID();
    if (!trajectoryNavigator->IsReferencePhoton(trackID)) {
      continue;
    }

    const MultiPhotonAnalysisHelpers::InteractionProcess process =
        MultiPhotonAnalysisHelpers::GetInteractionProcess(phantomHit->GetProcess());
    if (process == MultiPhotonAnalysisHelpers::InteractionProcess::Other) {
      continue;
    }

    MultiPhotonAnalysisHelpers::AccumulatePhantomTotals(
        phantomHit->GetProcess(), LocateVolumeName(phantomHit->GetPos()), totals[trackID]);
  }

  return totals;
}

EventContext BuildEventContext(
    const G4Event *event,
    GateRunManager *runManager,
    GateMultiPhotonTrajectoryNavigator *trajectoryNavigator) {
  EventContext context;
  context.event_id = event->GetEventID();
  context.run_id = runManager->GetCurrentRun()->GetRunID();

  GateSourceMgr *sourceMgr = GateSourceMgr::GetInstance();
  const std::vector<GateVSource *> &sourcesForThisEvent = sourceMgr->GetSourcesForThisEvent();
  if (!sourcesForThisEvent.empty()) {
    context.source_id = sourcesForThisEvent[0]->GetSourceID();
    context.source_vertex = trajectoryNavigator->FindSourcePosition();
  }

  return context;
}

std::vector<TimelineEntry> BuildTimelineForCrystalCollection(
    const std::vector<TimelineEntry> &basePhantomTimeline,
    GateHitsCollection *CHC) {
  std::vector<TimelineEntry> timeline = basePhantomTimeline;
  std::size_t sequence = timeline.size();

  const G4int NbHits = CHC->entries();
  timeline.reserve(basePhantomTimeline.size() + static_cast<std::size_t>(NbHits));
  for (G4int iHit = 0; iHit < NbHits; ++iHit) {
    GateHit *crystalHit = (*CHC)[iHit];
    if (!crystalHit) {
      continue;
    }

    // Hits whose gamma cannot be resolved are kept as well: they receive zeroed counters,
    // but eventID, runID and the remaining attributes are filled in like in GateAnalysis.
    TimelineEntry entry;
    entry.time = crystalHit->GetTime();
    entry.sequence = sequence++;
    entry.is_phantom = false;
    entry.track_id = crystalHit->GetTrackID();
    entry.crystal_hit = crystalHit;
    timeline.push_back(entry);
  }

  std::sort(timeline.begin(), timeline.end(), [](const TimelineEntry &a, const TimelineEntry &b) {
    if (a.time < b.time) {
      return true;
    }
    if (a.time > b.time) {
      return false;
    }
    return a.sequence < b.sequence;
  });

  return timeline;
}

void ProcessTimeline(
    std::vector<TimelineEntry> *timeline,
    GateMultiPhotonTrajectoryNavigator *trajectoryNavigator,
    const EventContext &context,
    const std::unordered_map<int, MultiPhotonAnalysisHelpers::PhantomStatistics> &phantomTotals,
    int septalNb,
    int legacyPhotonIDPolicy) {
  const MultiPhotonAnalysisHelpers::PhantomStatistics kNoPhantomStatistics;

  std::unordered_map<int, MultiPhotonAnalysisHelpers::RunningStatistics> runningStatsByPhotonTrackId;
  runningStatsByPhotonTrackId.reserve(timeline->size());

  for (std::size_t idx = 0; idx < timeline->size(); ++idx) {
    TimelineEntry &entry = (*timeline)[idx];

    // Counters are incremented only by hits of the reference photon itself (attribution by
    // exact track ID, like in GateAnalysis) and BEFORE the hit is written, so that the value
    // stored in the hit includes the current interaction.
    if (trajectoryNavigator->IsReferencePhoton(entry.track_id)) {
      MultiPhotonAnalysisHelpers::RunningStatistics &runningStats =
          runningStatsByPhotonTrackId[entry.track_id];
      if (entry.is_phantom) {
        if (entry.phantom_hit) {
          MultiPhotonAnalysisHelpers::AccumulatePhantom(entry.phantom_hit->GetProcess(), runningStats);
        }
      } else if (entry.crystal_hit) {
        MultiPhotonAnalysisHelpers::AccumulateCrystal(entry.crystal_hit->GetProcess(), runningStats);
      }
    }

    if (entry.is_phantom || !entry.crystal_hit || !entry.crystal_hit->GoodForAnalysis()) {
      continue;
    }

    GateHit *hit = entry.crystal_hit;

    // Which gamma the hit belongs to is resolved through the ancestry chain, so hits of
    // secondary tracks receive the counters of their parent gamma without incrementing them.
    const int ancestorPhoton = trajectoryNavigator->FindAncestorPhotonTrackID(entry.track_id);

    const MultiPhotonAnalysisHelpers::PhantomStatistics *phantomStats = &kNoPhantomStatistics;
    MultiPhotonAnalysisHelpers::RunningStatistics runningStats;
    if (ancestorPhoton != 0) {
      const std::unordered_map<int, MultiPhotonAnalysisHelpers::PhantomStatistics>::const_iterator
          phantom_it = phantomTotals.find(ancestorPhoton);
      if (phantom_it != phantomTotals.end()) {
        phantomStats = &(phantom_it->second);
      }

      const std::unordered_map<int, MultiPhotonAnalysisHelpers::RunningStatistics>::const_iterator
          running_it = runningStatsByPhotonTrackId.find(ancestorPhoton);
      if (running_it != runningStatsByPhotonTrackId.end()) {
        runningStats = running_it->second;
      }
    }

    hit->SetSourceID(context.source_id);
    hit->SetSourcePosition(context.source_vertex);
    hit->SetNPhantomCompton(phantomStats->compton);
    hit->SetNPhantomRayleigh(phantomStats->rayleigh);
    // Volume names are always written, also when no scattering happened - an empty string would
    // make the digitizer overwrite sourceID with -1 in the Singles and Coincidences trees.
    hit->SetComptonVolumeName(phantomStats->comptonVolumeName.c_str());
    hit->SetRayleighVolumeName(phantomStats->rayleighVolumeName.c_str());
    hit->SetPhotonID(legacyPhotonIDPolicy);
    hit->SetPrimaryID(trajectoryNavigator->FindPrimaryTrackID(entry.track_id));
    hit->SetEventID(context.event_id);
    hit->SetRunID(context.run_id);
    hit->SetNCrystalCompton(runningStats.crystalCompton);
    hit->SetNCrystalRayleigh(runningStats.crystalRayleigh);
    hit->SetNSeptal(septalNb);
    // nInteractions is intentionally filled only in the multiphoton analysis path.
    hit->SetNInteractions(runningStats.scatters);
  }
}

void RunDigitizersIfNeeded() {
  GateDigitizerMgr *digitizerMgr = GateDigitizerMgr::GetInstance();
  if (!digitizerMgr->m_alreadyRun) {
    if (digitizerMgr->m_recordSingles || digitizerMgr->m_recordCoincidences) {
      digitizerMgr->RunDigitizers();
      digitizerMgr->RunCoincidenceSorters();
      digitizerMgr->RunCoincidenceDigitizers();
    }
  }
}

//! Runs the digitizer chain when the enclosing scope is left, whichever path is taken.
//! GateAnalysis calls the digitizer outside of its "no trajectory container" branch, so
//! whether Singles and Coincidences are produced for an event does not depend on the
//! analysis being able to process that event. The guard gives the same guarantee here,
//! where the event is abandoned in more than one place.
class ScopedDigitizerRunner {
 public:
  ScopedDigitizerRunner() = default;
  ~ScopedDigitizerRunner() { RunDigitizersIfNeeded(); }

  ScopedDigitizerRunner(const ScopedDigitizerRunner &) = delete;
  ScopedDigitizerRunner &operator=(const ScopedDigitizerRunner &) = delete;
};

}  // namespace

GateMultiPhotonAnalysis::GateMultiPhotonAnalysis(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode)
    : GateVOutputModule(name, outputMgr, digiMode),
      m_trajectoryNavigator(new GateMultiPhotonTrajectoryNavigator()),
      m_messenger(new GateMultiPhotonAnalysisMessenger(this)),
      m_missingTrajectoryPolicy(kResilient),
      m_missingTrajectoryEventCount(0),
      m_missingTrajectoryWithHitsCount(0) {
  m_isEnabled = false;
  SetVerboseLevel(0);
}

GateMultiPhotonAnalysis::~GateMultiPhotonAnalysis() {
  delete m_messenger;
  m_messenger = 0;
  delete m_trajectoryNavigator;
  m_trajectoryNavigator = 0;
}

const G4String &GateMultiPhotonAnalysis::GiveNameOfFile() {
  m_noFileName = "  ";
  return m_noFileName;
}

void GateMultiPhotonAnalysis::RecordBeginOfAcquisition() {
  if (nVerboseLevel > 2) {
    G4cout << "GateMultiPhotonAnalysis::RecordBeginOfAcquisition" << G4endl;
  }
}

void GateMultiPhotonAnalysis::RecordEndOfAcquisition() {
  if (nVerboseLevel > 2) {
    G4cout << "GateMultiPhotonAnalysis::RecordEndOfAcquisition" << G4endl;
  }
}

void GateMultiPhotonAnalysis::RecordBeginOfRun(const G4Run *) {
  if (nVerboseLevel > 2) {
    G4cout << "GateMultiPhotonAnalysis::RecordBeginOfRun" << G4endl;
  }
}

void GateMultiPhotonAnalysis::RecordEndOfRun(const G4Run *) {
  if (nVerboseLevel > 2) {
    G4cout << "GateMultiPhotonAnalysis::RecordEndOfRun" << G4endl;
  }

  if (m_missingTrajectoryEventCount > 0) {
    G4cout
        << "[GateMultiPhotonAnalysis] Missing trajectory container in "
        << m_missingTrajectoryEventCount
        << " event(s), including "
        << m_missingTrajectoryWithHitsCount
        << " event(s) with processable hits. Policy="
        << GetMissingTrajectoryPolicyName()
        << G4endl;
  }
}

void GateMultiPhotonAnalysis::RecordBeginOfEvent(const G4Event *) {
  if (nVerboseLevel > 2) {
    G4cout << "GateMultiPhotonAnalysis::RecordBeginOfEvent" << G4endl;
  }
}

void GateMultiPhotonAnalysis::RecordEndOfEvent(const G4Event *event) {
  if (!event) {
    return;
  }

  GateRunManager *runManager = GateRunManager::GetRunManager();
  GateSteppingAction *steppingAction = (GateSteppingAction *)(runManager->GetUserSteppingAction());
  TrackingMode mode = steppingAction->GetMode();
  const int tracking_mode_code = static_cast<int>(mode);
  if (!IsTrackingModeSupported(tracking_mode_code)) {
    G4Exception(
        "GateMultiPhotonAnalysis::RecordEndOfEvent",
        "GateMultiPhotonAnalysisUnsupportedTrackingMode",
        FatalException,
        "GateMultiPhotonAnalysis cannot process the current tracking mode. "
        "Only TrackingMode::kBoth is supported, and continuing would skip the "
        "remaining end-of-event processing, including digitizer output.");
    return;
  }

  // From here on the event is digitized whatever happens to the analysis itself.
  ScopedDigitizerRunner digitizerRunner;

  // An event without any primary vertex carries no track, hence no hit and no trajectory
  // container. GateSourceMgr stops generating vertices once the time limit of the run is
  // reached, so the last event of every run looks exactly like this. There is nothing to
  // analyse and nothing anomalous about it, so it is skipped quietly instead of being
  // reported as a missing trajectory container.
  if (event->GetNumberOfPrimaryVertex() == 0) {
    return;
  }

  std::vector<GateHitsCollection *> CHC_vector = GetOutputMgr()->GetHitCollections();
  GatePhantomHitsCollection *PHC = GetOutputMgr()->GetPhantomHitCollection();

  G4TrajectoryContainer *trajectoryContainer = event->GetTrajectoryContainer();
  if (!trajectoryContainer) {
    ++m_missingTrajectoryEventCount;

    const bool hasProcessableHits = EventHasProcessableHits(CHC_vector, PHC);
    if (hasProcessableHits) {
      ++m_missingTrajectoryWithHitsCount;
    }

    G4int runID = -1;
    if (runManager->GetCurrentRun()) {
      runID = runManager->GetCurrentRun()->GetRunID();
    }
    const G4int eventID = event->GetEventID();

    G4String details = "GateMultiPhotonAnalysis missing trajectory container for run="
        + std::to_string(runID)
        + ", event="
        + std::to_string(eventID)
        + ". hasProcessableHits="
        + (hasProcessableHits ? "true" : "false")
        + ". Policy="
        + GetMissingTrajectoryPolicyName()
        + ".";

    if (m_missingTrajectoryPolicy == kStrict) {
      G4Exception(
          "GateMultiPhotonAnalysis::RecordEndOfEvent",
          "GateMultiPhotonAnalysisMissingTrajectoryContainer",
          FatalException,
          details.c_str());
      return;
    }

    G4Exception(
        "GateMultiPhotonAnalysis::RecordEndOfEvent",
        "GateMultiPhotonAnalysisMissingTrajectoryContainer",
        JustWarning,
        details.c_str());
    return;
  }

  m_trajectoryNavigator->SetTrajectoryContainer(trajectoryContainer);
  m_trajectoryNavigator->BuildIndex();
  std::vector<TimelineEntry> basePhantomTimeline = BuildBasePhantomTimeline(PHC);
  const std::unordered_map<int, MultiPhotonAnalysisHelpers::PhantomStatistics> phantomTotals =
      ComputePhantomTotals(PHC, m_trajectoryNavigator);
  const int septalNb = CountSeptalHits(PHC);
  const EventContext context = BuildEventContext(event, runManager, m_trajectoryNavigator);
  const int legacyPhotonIDPolicy = ResolveLegacyPhotonIDPolicy();

  for (size_t i = 0; i < CHC_vector.size(); ++i) {
    GateHitsCollection *CHC = CHC_vector[i];
    if (!CHC) {
      continue;
    }

    std::vector<TimelineEntry> timeline = BuildTimelineForCrystalCollection(basePhantomTimeline, CHC);
    ProcessTimeline(&timeline, m_trajectoryNavigator, context, phantomTotals, septalNb,
                    legacyPhotonIDPolicy);
  }
}

void GateMultiPhotonAnalysis::RecordStepWithVolume(const GateVVolume *, const G4Step *) {
  if (nVerboseLevel > 2) {
    G4cout << "GateMultiPhotonAnalysis::RecordStepWithVolume" << G4endl;
  }
}

void GateMultiPhotonAnalysis::SetVerboseLevel(G4int val) {
  nVerboseLevel = val;
  if (m_trajectoryNavigator) {
    m_trajectoryNavigator->SetVerboseLevel(val);
  }
}

bool GateMultiPhotonAnalysis::IsTrackingModeSupported(int tracking_mode_code) const {
  return tracking_mode_code == static_cast<int>(TrackingMode::kBoth);
}

int GateMultiPhotonAnalysis::ResolveLegacyPhotonIDPolicy() const {
  // photonID is a GateAnalysis-specific field: it indexes the two gammas of a back-to-back
  // annihilation (1 or 2, 0 when the hit comes from neither). There is no meaningful
  // generalization for an arbitrary number of reference photons, and the field is not
  // propagated to the Singles or Coincidences trees, so multi-photon analysis always writes 0.
  // The value therefore differs from GateAnalysis in the Hits tree - this is intentional.
  return 0;
}

void GateMultiPhotonAnalysis::SetMissingTrajectoryPolicy(MissingTrajectoryPolicy policy) {
  m_missingTrajectoryPolicy = policy;
}

void GateMultiPhotonAnalysis::SetMissingTrajectoryPolicyFromString(const G4String &policyName) {
  if (policyName == "strict") {
    m_missingTrajectoryPolicy = kStrict;
    return;
  }
  if (policyName == "resilient") {
    m_missingTrajectoryPolicy = kResilient;
    return;
  }

  G4String message = "Unsupported missing trajectory policy: " + policyName
      + ". Expected one of: strict resilient.";
  G4Exception(
      "GateMultiPhotonAnalysis::SetMissingTrajectoryPolicyFromString",
      "GateMultiPhotonAnalysisInvalidPolicy",
      JustWarning,
      message.c_str());
}

GateMultiPhotonAnalysis::MissingTrajectoryPolicy GateMultiPhotonAnalysis::GetMissingTrajectoryPolicy() const {
  return m_missingTrajectoryPolicy;
}

G4String GateMultiPhotonAnalysis::GetMissingTrajectoryPolicyName() const {
  if (m_missingTrajectoryPolicy == kStrict) {
    return "strict";
  }
  return "resilient";
}

