/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateComptonCameraActorMessenger
*/

#include "GateConfiguration.h"

#ifndef GATECOMPTONCAMERAACTORMESSENGER_HH
#define GATECOMPTONCAMERAACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"

#include "GateActorMessenger.hh"

class GateComptonCameraActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateComptonCameraActorMessenger : public GateActorMessenger
{
public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateComptonCameraActorMessenger(GateComptonCameraActor * v);
  /// Destructor
  virtual ~GateComptonCameraActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateComptonCameraActor * pActor;

  /// Command objects
  G4UIcmdWithABool          * pSaveHitsTree;
  G4UIcmdWithABool          *pSaveCoincidenceText;
  G4UIcmdWithABool          * pSaveSinglesText;

}; // end class GateComptonCameraActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATECOMPTONCAMERAACTORMESSENGER_HH */
//#endif
