/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*
  \class  GateEnergySpectrumActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
  andreas.resch@meduniwien.ac.at
*/

#ifndef GATEENERGYSPECTRUMACTORMESSENGER_HH
#define GATEENERGYSPECTRUMACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"

#include "GateActorMessenger.hh"

class GateEnergySpectrumActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateEnergySpectrumActorMessenger : public GateActorMessenger
{
public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateEnergySpectrumActorMessenger(GateEnergySpectrumActor * v);
  /// Destructor
  virtual ~GateEnergySpectrumActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateEnergySpectrumActor * pActor;

  /// Command objects
  G4UIcmdWithADoubleAndUnit * pEmaxCmd;
  G4UIcmdWithADoubleAndUnit * pEminCmd;
  G4UIcmdWithADoubleAndUnit * pLETminCmd;
  G4UIcmdWithADoubleAndUnit * pLETmaxCmd;
  G4UIcmdWithAnInteger      * pNLETBinsCmd;
  G4UIcmdWithADoubleAndUnit * pQminCmd;
  G4UIcmdWithADoubleAndUnit * pQmaxCmd;
  G4UIcmdWithAnInteger      * pNQBinsCmd;
  G4UIcmdWithAnInteger      * pNBinsCmd;
  G4UIcmdWithADoubleAndUnit * pEdepmaxCmd;
  G4UIcmdWithADoubleAndUnit * pEdepminCmd; 
  G4UIcmdWithAnInteger      * pEdepNBinsCmd;
  G4UIcmdWithABool          * pSaveAsText;
  G4UIcmdWithABool          * pSaveAsTextDiscreteEnergySpectrum;
  G4UIcmdWithABool          * pEnableLETSpectrumCmd;
  G4UIcmdWithABool          * pEnableLETFluenceSpectrumCmd;
  G4UIcmdWithABool          * pEnableLETtoMaterialFluenceSpectrumCmd;
  G4UIcmdWithAString * pSetOtherMaterialCmd;
  G4UIcmdWithABool          * pEnableQSpectrumCmd;
  G4UIcmdWithABool          * pEnableEnergySpectrumNbPartCmd;
  G4UIcmdWithABool          * pEnableEnergySpectrumFluenceCosCmd;
  G4UIcmdWithABool          * pEnableEnergySpectrumFluenceTrackCmd;
  G4UIcmdWithABool          * pEnableeEnergySpectrumEdepCmd;
  G4UIcmdWithABool          * pEnableEdepHistoCmd;
  G4UIcmdWithABool          * pEnableEdepTimeHistoCmd;
  G4UIcmdWithABool          * pEnableEdepTrackHistoCmd;
  G4UIcmdWithABool          * pEnableEdepStepHistoCmd;
  G4UIcmdWithABool          * pEnableElossHistoCmd;
  
  G4UIcmdWithABool          * pEnableLogBinningCMD;
  G4UIcmdWithABool          * pEnableEnergyPerUnitMassCMD;
  G4UIcmdWithABool          * pEnableRelativePrimEventsCMD;
  

}; // end class GateEnergySpectrumActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEENERGYSPECTRUMACTORMESSENGER_HH */
#endif
