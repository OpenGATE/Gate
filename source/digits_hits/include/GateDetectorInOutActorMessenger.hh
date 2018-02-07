/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class  GateDetectorInOutActorMessenger
  \author simon.rit@creatis.insa-lyon.fr
*/

#ifndef GATEDETECTORINOUTMESSENGER_HH
#define GATEDETECTORINOUTMESSENGER_HH

#include "GateActorMessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

class GateDetectorInOutActor;

//-----------------------------------------------------------------------------
class GateDetectorInOutActorMessenger : public GateActorMessenger
{
public:
  GateDetectorInOutActorMessenger(GateDetectorInOutActor* sensor);
  virtual ~GateDetectorInOutActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateDetectorInOutActor    * pDIOActor;
  G4UIcmdWithAString        * pSetOutputWindowNamesCmd;
  G4UIcmdWithABool          * pSetOutputInDataOnlyFlagCmd;
  G4UIcmdWithADoubleAndUnit * pSetMaxAngleCmd;
  G4UIcmdWithAnInteger      * pSetRRFactorCmd;
};
//-----------------------------------------------------------------------------

#endif /* end #define GATEDETECTORINOUTMESSENGER_HH */
