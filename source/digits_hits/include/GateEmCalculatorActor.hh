/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class GateEmCalculatorActor
  \author loic.grevillot@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
 */

#include "GateConfiguration.h"

#ifndef GATEEmCalculatorActor_HH
#define GATEEmCalculatorActor_HH

#include "GateVActor.hh"
#include "GateEmCalculatorActorMessenger.hh"
#include "GateActorManager.hh"
//#include "GateActorMessenger.hh"
#include "G4EmCalculator.hh"

//-----------------------------------------------------------------------------
/// \brief Actor displaying stopping powers
class GateEmCalculatorActor : public GateVActor
{
 public:

  virtual ~GateEmCalculatorActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateEmCalculatorActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Callbacks
//    virtual void BeginOfRunAction(const G4Run*);
//    virtual void BeginOfEventAction(const G4Event*);
//    virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
//    virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  // other methods
  void SetEnergy (double E) {mEnergy=E;}
  void SetParticleName (G4String Name) {mPartName=Name;}
  //Particle Properties If GenericIon
  void SetIonParameter(G4String ParticleParameters) {mParticleParameters=ParticleParameters;}
  //Specify how to define the particle type
  void SetIsGenericIon(bool IsGenericIon) {mIsGenericIon=IsGenericIon;}

  //-----------------------------------------------------------------------------
  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

protected:
  const G4ParticleDefinition* GetIonDefinition();

  double mEnergy;
  G4String mPartName;
  G4String mParticleParameters;
  bool mIsGenericIon;

  G4EmCalculator * emcalc;
  GateEmCalculatorActor(G4String name, G4int depth=0);
  GateEmCalculatorActorMessenger * pActorMessenger;
//  GateActorMessenger * pActor;
};

MAKE_AUTO_CREATOR_ACTOR(EmCalculatorActor,GateEmCalculatorActor)


#endif /* end #define GATEEmCalculatorActor_HH */
