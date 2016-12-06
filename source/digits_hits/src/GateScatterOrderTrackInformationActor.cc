/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

// Gate 
#include "GateScatterOrderTrackInformationActor.hh"
#include "GateMiscFunctions.hh"

// G4
#include <G4Event.hh>
#include <G4MaterialTable.hh>
#include <G4ParticleTable.hh>
#include <G4VEmProcess.hh>
#include <G4TransportationManager.hh>
#include <G4LivermoreComptonModel.hh>

#define TRY_AND_EXIT_ON_ITK_EXCEPTION(execFunc)                         \
  try                                                                   \
    {                                                                   \
    execFunc;                                                          \
    }                                                                   \
  catch( itk::ExceptionObject & err )                                   \
    {                                                                   \
    std::cerr << "ExceptionObject caught with " #execFunc << Gateendl; \
    std::cerr << err << Gateendl;                                      \
    exit(EXIT_FAILURE);                                                 \
    }

//-----------------------------------------------------------------------------

/// Constructors
GateScatterOrderTrackInformationActor::GateScatterOrderTrackInformationActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateScatterOrderTrackInformationActor() -- begin\n");
  pActorMessenger = new GateScatterOrderTrackInformationActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateScatterOrderTrackInformationActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateScatterOrderTrackInformationActor::~GateScatterOrderTrackInformationActor()
{
  delete pActorMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateScatterOrderTrackInformationActor::Construct()
{
  GateVActor::Construct();
  //  Callbacks
  EnableUserSteppingAction(true);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateScatterOrderTrackInformationActor::UserSteppingAction(const GateVVolume * v,
                                                               const G4Step * step)
{
  GateVActor::UserSteppingAction(v, step);
  // Create a track information and attach it to the track
     if(step->GetTrack()->GetUserInformation()==0)
     {
       GateScatterOrderTrackInformation* anInfo = new GateScatterOrderTrackInformation;
       step->GetTrack()->SetUserInformation(anInfo);
     }
  // We are only interested in EM processes. One should check post-step to know
  // what is going to happen, but pre-step is used afterward to get direction
  // and position.
  const G4VProcess *pr = step->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VEmProcess *process = dynamic_cast<const G4VEmProcess*>(pr);
  if(!process) return;

  //unsigned int order = 0;
  //G4String scatterProcess = "";

  if(process->GetProcessName() == G4String("Compton") ||
     process->GetProcessName() == G4String("compt")) {

    GateScatterOrderTrackInformation * scatterTracking = dynamic_cast<GateScatterOrderTrackInformation*>(step->GetTrack()->GetUserInformation());
    if(scatterTracking!=NULL)
    {
      scatterTracking->IncrementScatterOrder(step->GetTrack());
      //order = scatterTracking->GetScatterOrder();
      // Set process name
      scatterTracking->SetScatterProcess(step->GetTrack());
    }
  }
  else if(process->GetProcessName() == G4String("RayleighScattering") ||
          process->GetProcessName() == G4String("Rayl")) {

    GateScatterOrderTrackInformation * scatterTracking = dynamic_cast<GateScatterOrderTrackInformation*>(step->GetTrack()->GetUserInformation());
    if(scatterTracking!=NULL)
    {
      scatterTracking->IncrementScatterOrder(step->GetTrack());
      //order = scatterTracking->GetScatterOrder();
      // Set process name
      scatterTracking->SetScatterProcess(step->GetTrack());
    }
  }
  else if(process->GetProcessName() == G4String("PhotoElectric") ||
          process->GetProcessName() == G4String("phot")) {
    const G4TrackVector * list = step->GetSecondary();

    for(unsigned short i=0; i<(*list).size(); i++)
    {
      if((*list)[i]->GetUserInformation()==0 && (*list)[i]->GetDefinition()->GetParticleName()=="gamma")
      {
        //GateScatterOrderTrackInformation * fluoTracking = (*list)[i]->GetUserInformation()
        // Creation of gamma
        GateScatterOrderTrackInformation* fluoInfo = new GateScatterOrderTrackInformation;
        (*list)[i]->SetUserInformation(fluoInfo);
      }
      GateScatterOrderTrackInformation * fluoTracking = dynamic_cast<GateScatterOrderTrackInformation*>((*list)[i]->GetUserInformation());
      if(fluoTracking!=NULL)
      {
        fluoTracking->IncrementScatterOrder((*list)[i]);
        //order = fluoTracking->GetScatterOrder();
        // Set process name
        fluoTracking->SetScatterProcess(step->GetTrack());
      }
    }
  }
}

//-----------------------------------------------------------------------------
