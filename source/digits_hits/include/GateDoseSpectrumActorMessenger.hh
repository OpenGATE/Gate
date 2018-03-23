/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*
  \class  GateDoseSpectrumActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEDOSESPECTRUMACTORMESSENGER_HH
#define GATEDOSESPECTRUMACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateDoseSpectrumActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateDoseSpectrumActorMessenger : public GateActorMessenger
{
 public: 
  
  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateDoseSpectrumActorMessenger(GateDoseSpectrumActor * v);
  /// Destructor
  virtual ~GateDoseSpectrumActorMessenger();
    
  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateDoseSpectrumActor * pDoseSpectrumActor;
  G4UIcmdWithAString * pWriteDoseResponseCmd;
  G4UIcmdWithABool* pDosePrimaryOnlyCmd;
}; // end class GateDoseSpectrumActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEDOSESPECTRUMACTORMESSENGER_HH */
#endif
