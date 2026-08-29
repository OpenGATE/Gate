/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateMultiPhotonAnalysis.hh"

#include "GateActions.hh"
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
#include "G4Run.hh"
#include "G4TrajectoryContainer.hh"

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
  int ancestor_photon = 0;
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

std::vector<TimelineEntry> BuildBasePhantomTimeline(
    GatePhantomHitsCollection *PHC,
    GateMultiPhotonTrajectoryNavigator *trajectoryNavigator) {
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

    const G4int trackID = phantomHit->GetTrackID();
    const int ancestorPhoton = trajectoryNavigator->FindAncestorPhotonTrackID(trackID);
    if (ancestorPhoton == 0) {
      continue;
    }

    TimelineEntry entry;
    entry.time = phantomHit->GetTime();
    entry.sequence = sequence++;
    entry.is_phantom = true;
    entry.track_id = trackID;
    entry.ancestor_photon = ancestorPhoton;
    entry.phantom_hit = phantomHit;
    basePhantomTimeline.push_back(entry);
  }

  return basePhantomTimeline;
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
    GateHitsCollection *CHC,
    GateMultiPhotonTrajectoryNavigator *trajectoryNavigator) {
  std::vector<TimelineEntry> timeline = basePhantomTimeline;
  std::size_t sequence = timeline.size();

  const G4int NbHits = CHC->entries();
  timeline.reserve(basePhantomTimeline.size() + static_cast<std::size_t>(NbHits));
  for (G4int iHit = 0; iHit < NbHits; ++iHit) {
    GateHit *crystalHit = (*CHC)[iHit];
    if (!crystalHit) {
      continue;
    }

    const G4int trackID = crystalHit->GetTrackID();
    const int ancestorPhoton = trajectoryNavigator->FindAncestorPhotonTrackID(trackID);
    if (ancestorPhoton == 0) {
      continue;
    }

    TimelineEntry entry;
    entry.time = crystalHit->GetTime();
    entry.sequence = sequence++;
    entry.is_phantom = false;
    entry.track_id = trackID;
    entry.ancestor_photon = ancestorPhoton;
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
    int legacyPhotonIDPolicy) {
  std::unordered_map<int, MultiPhotonAnalysisHelpers::GammaStatistics> runningStatsByPhotonTrackId;
  runningStatsByPhotonTrackId.reserve(timeline->size());

  for (std::size_t idx = 0; idx < timeline->size(); ++idx) {
    TimelineEntry &entry = (*timeline)[idx];
    MultiPhotonAnalysisHelpers::GammaStatistics &runningStats = runningStatsByPhotonTrackId[entry.ancestor_photon];

    if (!entry.is_phantom && entry.crystal_hit && entry.crystal_hit->GoodForAnalysis()) {
      GateHit *hit = entry.crystal_hit;
      const int primaryID = trajectoryNavigator->FindPrimaryTrackID(entry.track_id);
      const int nInteractions = runningStats.phantomInteractions + runningStats.crystalInteractions;

      hit->SetSourceID(context.source_id);
      hit->SetSourcePosition(context.source_vertex);
      hit->SetNPhantomCompton(runningStats.phantomCompton);
      hit->SetNPhantomRayleigh(runningStats.phantomRayleigh);
      if (runningStats.comptonVolumeName != "NULL") {
        hit->SetComptonVolumeName(runningStats.comptonVolumeName.c_str());
      }
      if (runningStats.rayleighVolumeName != "NULL") {
        hit->SetRayleighVolumeName(runningStats.rayleighVolumeName.c_str());
      }
      hit->SetPhotonID(legacyPhotonIDPolicy);
      hit->SetPrimaryID(primaryID);
      hit->SetEventID(context.event_id);
      hit->SetRunID(context.run_id);
      hit->SetNCrystalCompton(runningStats.crystalCompton);
      hit->SetNCrystalRayleigh(runningStats.crystalRayleigh);
      // nInteractions is intentionally filled only in the multiphoton analysis path.
      hit->SetNInteractions(nInteractions);
    }

    if (entry.is_phantom) {
      if (!entry.phantom_hit) {
        continue;
      }

      MultiPhotonAnalysisHelpers::AccumulatePhantom(entry.phantom_hit->GetProcess(), runningStats);
      continue;
    }

    if (entry.crystal_hit) {
      MultiPhotonAnalysisHelpers::AccumulateCrystal(entry.crystal_hit->GetProcess(), runningStats);
    }
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

  // An event without any primary vertex carries no track, hence no hit and no trajectory
  // container. GateSourceMgr stops generating vertices once the time limit of the run is
  // reached, so the last event of every run looks exactly like this. There is nothing to
  // analyse and nothing anomalous about it, so it is skipped quietly instead of being
  // reported as a missing trajectory container.
  if (event->GetNumberOfPrimaryVertex() == 0) {
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
  std::vector<TimelineEntry> basePhantomTimeline = BuildBasePhantomTimeline(PHC, m_trajectoryNavigator);
  const EventContext context = BuildEventContext(event, runManager, m_trajectoryNavigator);
  const int legacyPhotonIDPolicy = ResolveLegacyPhotonIDPolicy();

  for (size_t i = 0; i < CHC_vector.size(); ++i) {
    GateHitsCollection *CHC = CHC_vector[i];
    if (!CHC) {
      continue;
    }

    std::vector<TimelineEntry> timeline = BuildTimelineForCrystalCollection(
        basePhantomTimeline,
        CHC,
        m_trajectoryNavigator);
    ProcessTimeline(&timeline, m_trajectoryNavigator, context, legacyPhotonIDPolicy);
  }

  RunDigitizersIfNeeded();
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

int GateMultiPhotonAnalysis::ResolveLegacyPhotonIDPolicy() const { return 0; }

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

