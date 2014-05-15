/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

//-----------------------------------------------------------------------------
/// \class GateScatterOrderTrackInformationActor
//-----------------------------------------------------------------------------

#include "GateConfiguration.h"

#ifndef GATESCATTERORDERTRACKINFORMATIONACTOR_HH
#define GATESCATTERORDERTRACKINFORMATIONACTOR_HH

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
#include "GateScatterOrderTrackInformationActorMessenger.hh"

// Geant4
#include <G4VEMDataSet.hh>
#include <G4EmCalculator.hh>
#include <G4VDataSetAlgorithm.hh>
#include <G4LivermoreComptonModel.hh>
#include <G4LogLogInterpolation.hh>
#include <G4CompositeEMDataSet.hh>

//-----------------------------------------------------------------------------

class GateScatterOrderTrackInformation:
    public G4VUserTrackInformation
{
public:


  GateScatterOrderTrackInformation():order(0) {}
  ~GateScatterOrderTrackInformation() {}

  inline void IncrementScatterOrder(const G4Track *track)
  {
    if(track->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()!=(const G4String)"Transportation")
      order++;
  }

  virtual inline G4int GetScatterOrder(){ return order; }

protected:
  G4int order;
};

//-----------------------------------------------------------------------------
class GateScatterOrderTrackInformationActorMessenger;
class GateScatterOrderTrackInformationActor : public GateVActor
{
public:

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateScatterOrderTrackInformationActor)

  GateScatterOrderTrackInformationActor(G4String name, G4int depth=0);
  ~GateScatterOrderTrackInformationActor();

  // Constructs the actor
  virtual void Construct();
  virtual void ResetData(){};

  // Callbacks
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

protected:
  GateScatterOrderTrackInformationActorMessenger * pActorMessenger;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(ScatterOrderTrackInformationActor, GateScatterOrderTrackInformationActor)


#endif /* end #define GATESCATTERORDERTRACKINFORMATIONACTOR_HH */
