/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

//-----------------------------------------------------------------------------
/// \class GateProtonNuclearInformationActor
//-----------------------------------------------------------------------------

#include "GateConfiguration.h"

#ifndef GATEPROTONNUCLEARINFORMATIONACTOR_HH
#define GATEPROTONNUCLEARINFORMATIONACTOR_HH

#include "globals.hh"
#include "G4String.hh"
#include <iomanip>   
#include <vector>

// Gate 
#include "GateVActor.hh"
#include "GateImage.hh"
#include "GateSourceMgr.hh"
#include "GateVImageVolume.hh"
#include <G4VEmProcess.hh>
#include "GateProtonNuclearInformationActorMessenger.hh"

// Geant4
//#include <G4VEMDataSet.hh>
//#include <G4EmCalculator.hh>
//#include <G4VDataSetAlgorithm.hh>
//#include <G4LivermoreComptonModel.hh>
//#include <G4LogLogInterpolation.hh>
//#include <G4CompositeEMDataSet.hh>

//-----------------------------------------------------------------------------

class GateProtonNuclearInformation:
    public G4VUserTrackInformation
{
public:
  GateProtonNuclearInformation():mOrder(0) {}
  ~GateProtonNuclearInformation() {}

  inline void IncrementScatterOrder(const G4Track *track)
  {
    // New particle
    if(track->GetStep()==NULL)
      mOrder++;
    // Same particle
    else if(track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()!=(const G4String)"Transportation")
      mOrder++;
  }

  inline void SetScatterProcess(const G4Track *track)
  {
    mProcess = track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
  }

  virtual inline G4int GetScatterOrder(){ return mOrder; }
  virtual inline G4String GetScatterProcess(){ return mProcess; }
  //virtual inline G4int SetScatterOrder(G4int _order){ return order=_order; }

protected:
  G4int    mOrder;
  G4String mProcess;

};

//-----------------------------------------------------------------------------
class GateProtonNuclearInformationActorMessenger;
class GateProtonNuclearInformationActor : public GateVActor
{
public:
  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateProtonNuclearInformationActor)

  GateProtonNuclearInformationActor(G4String name, G4int depth=0);
  ~GateProtonNuclearInformationActor();

  // Constructs the actor
  virtual void Construct();
  virtual void ResetData(){};

  // Callbacks
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

protected:
  GateProtonNuclearInformationActorMessenger * pActorMessenger;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(ProtonNuclearInformationActor, GateProtonNuclearInformationActor)

#endif /* end #define GateProtonNuclearInformationActor_HH */
