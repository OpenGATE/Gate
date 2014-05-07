/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
  \class  GateImageActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEVIMAGEACTORMESSENGER_HH
#define GATEVIMAGEACTORMESSENGER_HH

#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateActorMessenger.hh"

class GateVImageActor;

//-----------------------------------------------------------------------------
/// \brief Messenger of GateVImageActor
class GateImageActorMessenger : public GateActorMessenger
{
 public:

  //-----------------------------------------------------------------------------
  /// Constructor with pointer on the associated sensor
  GateImageActorMessenger(GateVImageActor * v);
  /// Destructor
  virtual ~GateImageActorMessenger();

  /// Command processing callback
  virtual void SetNewValue(G4UIcommand*, G4String);
  void BuildCommands(G4String base);

protected:

  /// Associated sensor
  GateVImageActor * pImageActor;

  /// Command objects
  G4UIcmdWith3VectorAndUnit * pVoxelSizeCmd;
  G4UIcmdWith3Vector        * pResolutionCmd;
  G4UIcmdWithAString        * pStepHitTypeCmd;
  G4UIcmdWith3VectorAndUnit * pHalfSizeCmd;
  G4UIcmdWith3VectorAndUnit * pSizeCmd;
  G4UIcmdWith3VectorAndUnit * pPositionCmd;

}; // end class GateImageActorMessenger
//-----------------------------------------------------------------------------

#endif /* end #define GATEVIMAGEACTORMESSENGER_HH */
