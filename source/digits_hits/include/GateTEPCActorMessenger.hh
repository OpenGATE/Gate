/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*!
  \class  GateTEPCActorMessenger
  \author smekens@clermont.in2p3.fr
*/

#ifndef GATETEPCACTORMESSENGER_HH
#define GATETEPCACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"

#include "GateActorMessenger.hh"

class GateTEPCActor;

//-----------------------------------------------------------------------------
/// \brief Messenger GateTEPCActor
class GateTEPCActorMessenger : public GateActorMessenger
{
public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateTEPCActorMessenger(GateTEPCActor * v);
  /// Destructor
  virtual ~GateTEPCActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateTEPCActor * pActor;

  G4UIcmdWithADoubleAndUnit * pPressureCmd;
  G4UIcmdWithADoubleAndUnit * pEmaxCmd;
  G4UIcmdWithADoubleAndUnit * pEminCmd;
  G4UIcmdWithAnInteger      * pEBinNumberCmd;
  G4UIcmdWithABool          * pELogscaleCmd;
  G4UIcmdWithAnInteger      * pENOrdersCmd;
  G4UIcmdWithABool          * pNormByEventCmd;
  G4UIcmdWithABool          * pSaveAsTextCmd;
  
}; // end class GateTEPCActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATETEPCACTORMESSENGER_HH */
#endif
