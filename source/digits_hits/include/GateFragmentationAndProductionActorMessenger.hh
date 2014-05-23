/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateFragmentationAndProductionActorMessenger
  \author pierre.gueth@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEFRAGMENTATIONANDPRODUCTIONACTORMESSENGER_HH
#define GATEFRAGMENTATIONANDPRODUCTIONACTORMESSENGER_HH

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateFragmentationAndProductionActor;

//-----------------------------------------------------------------------------
class GateFragmentationAndProductionActorMessenger : public GateActorMessenger
{
 public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateFragmentationAndProductionActorMessenger(GateFragmentationAndProductionActor * v);
  /// Destructor
  virtual ~GateFragmentationAndProductionActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);

  /// Associated sensor
  GateFragmentationAndProductionActor * pActor;

  /// Command objects
  G4UIcmdWithAnInteger * pNBinsCmd;
  //G4UIcmdWithADoubleAndUnit * pEmaxCmd;

}; // end class GateFragmentationAndProductionActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEFRAGMENTATIONANDPRODUCTIONACTORMESSENGER_HH */
#endif
