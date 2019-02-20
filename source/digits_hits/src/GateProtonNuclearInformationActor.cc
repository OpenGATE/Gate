/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

// Gate 
#include "GateProtonNuclearInformationActor.hh"
#include "GateMiscFunctions.hh"

// G4
#include <G4Event.hh>
#include <G4ParticleTable.hh>
#include <G4TransportationManager.hh>
#include "G4HadronicProcessStore.hh"
#include "G4HadronicProcess.hh"

/// Constructors
GateProtonNuclearInformationActor::GateProtonNuclearInformationActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateProtonNuclearInformationActor() -- begin"<<G4endl);
  pActorMessenger = new GateProtonNuclearInformationActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateProtonNuclearInformationActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateProtonNuclearInformationActor::~GateProtonNuclearInformationActor()
{
  delete pActorMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateProtonNuclearInformationActor::Construct()
{
  GateVActor::Construct();
  //  Callbacks
  EnableUserSteppingAction(true);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateProtonNuclearInformationActor::UserSteppingAction(const GateVVolume * v,
                                                               const G4Step * step)
{
  GateVActor::UserSteppingAction(v, step);
  // Create a track information and attach it to the track
     if(step->GetTrack()->GetUserInformation()==0)
     {
       GateProtonNuclearInformation* anInfo = new GateProtonNuclearInformation;
       step->GetTrack()->SetUserInformation(anInfo);
     }
  // One should check post-step to know
  // what is going to happen, but pre-step is used afterward to get direction
  // and position.
  const G4VProcess *pr = step->GetPostStepPoint()->GetProcessDefinedStep();
  const G4HadronicProcess *process = dynamic_cast<const G4HadronicProcess*>(pr);
  if(!process) return;

  if(process->GetProcessName() == G4String("hadElastic")) {

    GateProtonNuclearInformation * scatterTracking = dynamic_cast<GateProtonNuclearInformation*>(step->GetTrack()->GetUserInformation());
    if(scatterTracking!=NULL)
    {
      scatterTracking->IncrementScatterOrder(step->GetTrack());
      // Set process name
      scatterTracking->SetScatterProcess(step->GetTrack());
    }
  }
  else if(process->GetProcessName() == G4String("protonInelastic")) {

    GateProtonNuclearInformation * scatterTracking = dynamic_cast<GateProtonNuclearInformation*>(step->GetTrack()->GetUserInformation());
    if(scatterTracking!=NULL)
    {
      scatterTracking->IncrementScatterOrder(step->GetTrack());
      // Set process name
      scatterTracking->SetScatterProcess(step->GetTrack());
    }
  }
}

//-----------------------------------------------------------------------------
