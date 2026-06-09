/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GateMultiPhotonAnalysisMessenger_h
#define GateMultiPhotonAnalysisMessenger_h 1

#include "GateOutputModuleMessenger.hh"

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About: UI messenger for configuring and controlling the multi-photon analysis output module.
 **/

class G4UIcmdWithAString;

class GateMultiPhotonAnalysis;

/**
 * @brief Messenger for GateMultiPhotonAnalysis output module.
 *
 * The messenger inherits standard output-module commands (`enable`, `disable`,
 * `verbose`, `describe`) and adds `setMissingTrajectoryPolicy` for configuring
 * how missing trajectory containers are handled.
 */
class GateMultiPhotonAnalysisMessenger : public GateOutputModuleMessenger {
 public:
  /**
   * @brief Creates a messenger bound to GateMultiPhotonAnalysis.
   *
   * Args:
   *   gateMultiPhotonAnalysis: Owner module.
   */
  explicit GateMultiPhotonAnalysisMessenger(GateMultiPhotonAnalysis *gateMultiPhotonAnalysis);

  ~GateMultiPhotonAnalysisMessenger();

  /**
   * @brief Handles UI command updates.
   *
   * Args:
   *   command: UI command.
   *   newValue: Command value.
   */
  virtual void SetNewValue(G4UIcommand *command, G4String newValue);

 protected:
   /** @brief Bound multi-photon analysis module instance. */
  GateMultiPhotonAnalysis *m_gateMultiPhotonAnalysis;

   /** @brief UI command for setting missing trajectory policy. */
   G4UIcmdWithAString *m_setMissingTrajectoryPolicyCmd;
};

#endif
