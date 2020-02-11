/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateTrackLengthActorMessenger
  \author anders.garpebring@gmail.com
*/

#ifndef GATETLFLUENCEDISTRIBUTIONACTORMESSENGER_HH
#define GATETLFLUENCEDISTRIBUTIONACTORMESSENGER_HH

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateTLFluenceDistributionActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateTLFluenceDistributionActorMessenger : public GateActorMessenger
{
 public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateTLFluenceDistributionActorMessenger(GateTLFluenceDistributionActor * v);
  /// Destructor
  virtual ~GateTLFluenceDistributionActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateTLFluenceDistributionActor * pActor;

  /// Command objects
  G4UIcmdWithADoubleAndUnit * pEnergyMaxCmd;
  G4UIcmdWithADoubleAndUnit * pEnergyMinCmd;
  G4UIcmdWithADoubleAndUnit * pThetaMaxCmd;
  G4UIcmdWithADoubleAndUnit * pThetaMinCmd;
  G4UIcmdWithADoubleAndUnit * pPhiMaxCmd;
  G4UIcmdWithADoubleAndUnit * pPhiMinCmd;
  
  G4UIcmdWithAnInteger * pEnergyNBinsCmd;
  G4UIcmdWithAnInteger * pThetaNBinsCmd;
  G4UIcmdWithAnInteger * pPhiNBinsCmd;
  
  
  G4UIcmdWithABool * pEnergyEnableCmd;
  G4UIcmdWithABool * pThetaEnableCmd;
  G4UIcmdWithABool * pPhiEnableCmd;
  
  G4UIcmdWithAString *pAsciiFileCmd;
  

}; // end class GateTLFluenceDistributionActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATETLFLUENCEDISTRIBUTIONACTORMESSENGER_HH */
#endif
