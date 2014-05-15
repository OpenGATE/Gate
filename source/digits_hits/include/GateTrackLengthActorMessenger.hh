/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateTrackLengthActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATETRACKLENGTHACTORMESSENGER_HH
#define GATETRACKLENGTHACTORMESSENGER_HH

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateActorMessenger.hh"

class GateTrackLengthActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateTrackLengthActorMessenger : public GateActorMessenger
{
 public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateTrackLengthActorMessenger(GateTrackLengthActor * v);
  /// Destructor
  virtual ~GateTrackLengthActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateTrackLengthActor * pActor;

  /// Command objects
  G4UIcmdWithADoubleAndUnit * pLmaxCmd;
  G4UIcmdWithADoubleAndUnit * pLminCmd;
  G4UIcmdWithAnInteger * pNBinsCmd;

}; // end class GateTrackLengthActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATETRACKLENGTHACTORMESSENGER_HH */
#endif
