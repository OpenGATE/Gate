/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GateMultiPhotonAnalysis_h
#define GateMultiPhotonAnalysis_h

#include "GateMultiPhotonAnalysisHelpers.hh"
#include "GateVOutputModule.hh"

#include <vector>

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About: Multi-photon analysis output-module interface with event processing callbacks and trajectory-missing policies.
 **/

class GateVVolume;
class GateMultiPhotonTrajectoryNavigator;
class GateMultiPhotonAnalysisMessenger;

/**
 * @brief Multi-photon variant of event-end hit analysis.
 *
 * The class mirrors the legacy GateAnalysis role while removing the hardcoded
 * two-photon mapping assumption. It aggregates per-photon interaction counters
 * and assigns resolved values to crystal hits.
 *
 * Notes:
 *   - Iteration 1 supports TrackingMode::kBoth only.
 *   - Tracker/detector split mode handling is explicitly deferred.
 */
class GateMultiPhotonAnalysis : public GateVOutputModule {
 public:
  /** @brief Policy for handling missing trajectory containers. */
  enum MissingTrajectoryPolicy {
    kStrict,
    kResilient
  };

  /**
   * @brief Constructs the output module.
   *
   * Args:
   *   name: Output module name.
   *   outputMgr: Output manager owner.
   *   digiMode: Runtime/offline mode.
   */
  GateMultiPhotonAnalysis(const G4String &name, GateOutputMgr *outputMgr, DigiMode digiMode);

  virtual ~GateMultiPhotonAnalysis();

  /**
   * @brief Returns module file name placeholder.
   *
   * Returns:
   *   Dummy name since this module does not own a file.
   */
  const G4String &GiveNameOfFile();

  /** @brief Acquisition-begin callback. */
  void RecordBeginOfAcquisition();
  /** @brief Acquisition-end callback. */
  void RecordEndOfAcquisition();
  /** @brief Run-begin callback. */
  void RecordBeginOfRun(const G4Run *);
  /** @brief Run-end callback. */
  void RecordEndOfRun(const G4Run *);
  /** @brief Event-begin callback. */
  void RecordBeginOfEvent(const G4Event *);

  /**
   * @brief Performs event-end multi-photon aggregation and hit assignment.
   *
   * Args:
   *   event: Event carrying trajectory and hit collections.
   */
  void RecordEndOfEvent(const G4Event *event);

  /**
   * @brief Step callback (unused by this module).
   *
   * Args:
   *   v: Current volume.
   *   step: Current step.
   */
  void RecordStepWithVolume(const GateVVolume *v, const G4Step *step);

  /** @brief Voxel callback (not used). */
  void RecordVoxels(GateVGeometryVoxelStore *) {}

  /**
   * @brief Sets module and navigator verbosity.
   *
   * Args:
   *   val: Verbose level.
   */
  virtual void SetVerboseLevel(G4int val);

  /**
   * @brief Sets policy for missing trajectory container handling.
   *
   * Args:
   *   policy: Desired behavior policy.
   */
  void SetMissingTrajectoryPolicy(MissingTrajectoryPolicy policy);

  /**
   * @brief Parses and applies missing trajectory policy from text.
   *
   * Args:
   *   policyName: Policy name (`strict` or `resilient`).
   */
  void SetMissingTrajectoryPolicyFromString(const G4String &policyName);

  /**
   * @brief Returns current missing trajectory policy.
   *
   * Returns:
   *   Active policy value.
   */
  MissingTrajectoryPolicy GetMissingTrajectoryPolicy() const;

  /**
   * @brief Returns current missing trajectory policy as string.
   *
   * Returns:
   *   `strict` or `resilient`.
   */
  G4String GetMissingTrajectoryPolicyName() const;

 private:
   /**
    * @brief Checks whether current tracking mode is supported.
    *
    * Args:
    *   tracking_mode_code: Integer value of TrackingMode.
    *
    * Returns:
    *   True when event processing should continue for the current mode.
    */
   bool IsTrackingModeSupported(int tracking_mode_code) const;

  /**
   * @brief Returns policy value for legacy photonID field.
   *
   * Returns:
   *   Always 0 in multiphoton mode (explicit policy for iteration 1).
   */
  int ResolveLegacyPhotonIDPolicy() const;

  GateMultiPhotonTrajectoryNavigator *m_trajectoryNavigator;
   GateMultiPhotonAnalysisMessenger *m_messenger;
  G4String m_noFileName;
  MissingTrajectoryPolicy m_missingTrajectoryPolicy;
  G4int m_missingTrajectoryEventCount;
  G4int m_missingTrajectoryWithHitsCount;
};

#endif
