/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateMultiPhotonTrajectoryNavigator.hh"

#include "GateMultiPhotonTrajectoryNavigatorHelpers.hh"

#include "G4ThreeVector.hh"
#include "G4Trajectory.hh"
#include "G4TrajectoryContainer.hh"

#include "G4ios.hh"

GateMultiPhotonTrajectoryNavigator::GateMultiPhotonTrajectoryNavigator()
    : m_tc(0),
      m_positronTrackID(0),
      m_ionID(0),
      m_verboseLevel(0),
      m_hasSourcePosition(false),
      m_sourceX(0.0),
      m_sourceY(0.0),
      m_sourceZ(0.0) {}

GateMultiPhotonTrajectoryNavigator::~GateMultiPhotonTrajectoryNavigator() {}

void GateMultiPhotonTrajectoryNavigator::SetTrajectoryContainer(G4TrajectoryContainer *trajectoryContainer) {
  m_tc = trajectoryContainer;
  Reset();
}

void GateMultiPhotonTrajectoryNavigator::BuildIndex() {
  Reset();

  if (!m_tc) {
    if (m_verboseLevel > 0) {
      G4cout << "GateMultiPhotonTrajectoryNavigator::BuildIndex: WARNING null trajectory container" << G4endl;
    }
    return;
  }

  const int n_trajectories = m_tc->entries();
  const std::size_t expected_track_count = static_cast<std::size_t>(n_trajectories);
  m_parentByTrack.reserve(expected_track_count);
  m_pdgByTrack.reserve(expected_track_count);
  m_chargeByTrack.reserve(expected_track_count);
  m_referencePhotonTrackIDs.reserve(expected_track_count);
  m_ancestorPhotonCache.reserve(expected_track_count);
  m_primaryCache.reserve(expected_track_count);

  for (int i = 0; i < n_trajectories; ++i) {
    G4Trajectory *trj = (G4Trajectory *)((*m_tc)[i]);
    if (!trj) {
      continue;
    }

    const int track_id = trj->GetTrackID();
    m_parentByTrack[track_id] = trj->GetParentID();
    m_pdgByTrack[track_id] = trj->GetPDGEncoding();
    m_chargeByTrack[track_id] = trj->GetCharge();

    if (track_id == 1 && trj->GetPointEntries() > 0) {
      const G4ThreeVector p = ((G4TrajectoryPoint *)(trj->GetPoint(0)))->GetPosition();
      m_sourceX = p.x();
      m_sourceY = p.y();
      m_sourceZ = p.z();
      m_hasSourcePosition = true;
    }
  }

  for (std::unordered_map<int, double>::const_iterator it = m_chargeByTrack.begin(); it != m_chargeByTrack.end(); ++it) {
    if (it->second > 2.0) {
      m_ionID = 1;
      break;
    }
  }

  for (std::unordered_map<int, int>::const_iterator it = m_pdgByTrack.begin(); it != m_pdgByTrack.end(); ++it) {
    const int track_id = it->first;
    const int pdg = it->second;
    const std::unordered_map<int, int>::const_iterator parent_it = m_parentByTrack.find(track_id);
    if (parent_it == m_parentByTrack.end()) {
      continue;
    }
    if (parent_it->second == m_ionID && pdg == kPositronPDG) {
      m_positronTrackID = track_id;
      break;
    }
  }

  BuildReferencePhotonSet();
}

G4ThreeVector GateMultiPhotonTrajectoryNavigator::FindSourcePosition() const {
  if (!m_hasSourcePosition) {
    return G4ThreeVector();
  }
  return G4ThreeVector(m_sourceX, m_sourceY, m_sourceZ);
}

std::vector<int> GateMultiPhotonTrajectoryNavigator::FindReferencePhotonTrackIDs() const {
  std::vector<int> out;
  out.reserve(m_referencePhotonTrackIDs.size());
  for (std::unordered_set<int>::const_iterator it = m_referencePhotonTrackIDs.begin(); it != m_referencePhotonTrackIDs.end(); ++it) {
    out.push_back(*it);
  }
  return out;
}

int GateMultiPhotonTrajectoryNavigator::FindAncestorPhotonTrackID(int trackID) const {
  return MultiphotonTrajectoryResolver::ResolveAncestorPhotonTrackID(
      trackID,
      m_parentByTrack,
      m_referencePhotonTrackIDs,
      &m_ancestorPhotonCache);
}

int GateMultiPhotonTrajectoryNavigator::FindPrimaryTrackID(int trackID) const {
  return MultiphotonTrajectoryResolver::ResolvePrimaryTrackID(trackID, m_parentByTrack, &m_primaryCache);
}

void GateMultiPhotonTrajectoryNavigator::SetVerboseLevel(int v) { m_verboseLevel = v; }

void GateMultiPhotonTrajectoryNavigator::Reset() {
  m_parentByTrack.clear();
  m_pdgByTrack.clear();
  m_chargeByTrack.clear();
  m_referencePhotonTrackIDs.clear();
  m_ancestorPhotonCache.clear();
  m_primaryCache.clear();
  m_positronTrackID = 0;
  m_ionID = 0;
  m_hasSourcePosition = false;
  m_sourceX = 0.0;
  m_sourceY = 0.0;
  m_sourceZ = 0.0;
}

void GateMultiPhotonTrajectoryNavigator::BuildReferencePhotonSet() {
  for (std::unordered_map<int, int>::const_iterator it = m_pdgByTrack.begin(); it != m_pdgByTrack.end(); ++it) {
    const int track_id = it->first;
    const int pdg = it->second;
    if (pdg != kPhotonPDG) {
      continue;
    }

    const std::unordered_map<int, int>::const_iterator parent_it = m_parentByTrack.find(track_id);
    if (parent_it == m_parentByTrack.end()) {
      continue;
    }

    const int parent_id = parent_it->second;

    if (m_positronTrackID != 0) {
      if (parent_id == m_positronTrackID) {
        m_referencePhotonTrackIDs.insert(track_id);
      }
    } else {
      bool is_reference_photon = false;
      if (parent_id == 0) {
        is_reference_photon = true;
      } else if (m_ionID != 0 && parent_id == m_ionID) {
        is_reference_photon = true;
      } else {
        const std::unordered_map<int, int>::const_iterator parent_pdg_it = m_pdgByTrack.find(parent_id);
        if (parent_pdg_it != m_pdgByTrack.end() && parent_pdg_it->second == kPositronPDG) {
          is_reference_photon = true;
        }
      }

      if (is_reference_photon) {
        m_referencePhotonTrackIDs.insert(track_id);
      }
    }
  }
}
