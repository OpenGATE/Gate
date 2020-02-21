/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATETOSUMMARYMESSENGER_H
#define GATETOSUMMARYMESSENGER_H 1

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_FILE

#include "GateOutputModuleMessenger.hh"
#include "GateToSummary.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithoutParameter;

//--------------------------------------------------------------------------------
class GateToSummaryMessenger: public GateOutputModuleMessenger
{
public:
  GateToSummaryMessenger(GateToSummary* gateToASCII);
  ~GateToSummaryMessenger();

  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateToSummary* m_gateToSummary;
  G4UIcmdWithoutParameter* ResetCmd;
  G4UIcmdWithAString* SetFileNameCmd;
  G4UIcmdWithAString* m_addCollectionCmd;

};
//--------------------------------------------------------------------------------

#endif
#endif
