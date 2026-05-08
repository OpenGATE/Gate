/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GateMultiPhotonTrajectoryNavigator_h
#define GateMultiPhotonTrajectoryNavigator_h

#include "G4ThreeVector.hh"

#include <unordered_map>
#include <unordered_set>
#include <vector>

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About: Event-level trajectory index and ancestry resolution API for multi-photon analysis.
 **/

class G4TrajectoryContainer;

/**
 * @brief Resolves multi-photon trajectory ancestry for one event.
 *
 * This class builds per-event lookup tables that map track IDs to parent IDs
 * and provides cached ancestry queries for ancestor photon and primary track
 * resolution.
 */
class GateMultiPhotonTrajectoryNavigator {
 public:
  /**
   * @brief Constructs an empty event-scoped navigator.
   */
  GateMultiPhotonTrajectoryNavigator();

  virtual ~GateMultiPhotonTrajectoryNavigator();

  /**
   * @brief Sets trajectory container for current event and resets caches.
   *
   * Args:
   *   trajectoryContainer: Event trajectory container.
   */
  void SetTrajectoryContainer(G4TrajectoryContainer *trajectoryContainer);

  /**
   * @brief Builds all lookup indices for current trajectory container.
   */
  void BuildIndex();

  /**
   * @brief Returns source position inferred from source track.
   *
   * Returns:
   *   Source position when available, or default vector otherwise.
   */
  G4ThreeVector FindSourcePosition() const;

  /**
   * @brief Returns list of reference photon track IDs for current event.
   *
   * Returns:
   *   Reference photon track identifiers.
   */
  std::vector<int> FindReferencePhotonTrackIDs() const;

  /**
   * @brief Resolves ancestor reference photon for a track.
   *
   * Args:
   *   trackID: Track identifier to resolve.
   *
   * Returns:
   *   Ancestor reference photon track ID, or 0 if unresolved.
   */
  int FindAncestorPhotonTrackID(int trackID) const;

  /**
   * @brief Resolves primary track for a track.
   *
   * Args:
   *   trackID: Track identifier to resolve.
   *
   * Returns:
   *   Primary track ID, or 0 if unresolved.
   */
  int FindPrimaryTrackID(int trackID) const;

  /**
   * @brief Sets verbose level used for diagnostic warnings.
   *
   * Args:
   *   v: Verbose level.
   */
  void SetVerboseLevel(int v);

 private:
  /**
   * @brief Clears all internal caches and indices.
   */
  void Reset();

  /**
   * @brief Builds reference photon set according to legacy-compatible policy.
   */
  void BuildReferencePhotonSet();

   /** @brief PDG code for positron. */
   const int kPositronPDG = -11;

   /** @brief PDG code for photon. */
   const int kPhotonPDG = 22;

  G4TrajectoryContainer *m_tc;

  std::unordered_map<int, int> m_parentByTrack;
  std::unordered_map<int, int> m_pdgByTrack;
  std::unordered_map<int, double> m_chargeByTrack;
  std::unordered_set<int> m_referencePhotonTrackIDs;

  mutable std::unordered_map<int, int> m_ancestorPhotonCache;
  mutable std::unordered_map<int, int> m_primaryCache;

  int m_positronTrackID;
  int m_ionID;
  int m_verboseLevel;

  bool m_hasSourcePosition;
  double m_sourceX;
  double m_sourceY;
  double m_sourceZ;
};

#endif
