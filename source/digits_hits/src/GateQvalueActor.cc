/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateQvalueActor :
  \brief
*/


#include "GateQvalueActor.hh"
#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"
//-----------------------------------------------------------------------------
GateQvalueActor::GateQvalueActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {

  mCurrentEvent=-1;
  mNSec=0;
  pMessenger = new GateImageActorMessenger(this);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateQvalueActor::~GateQvalueActor()  {

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateQvalueActor::Construct() {
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  // Output Filename
  SetOriginTransformAndFlagToImage(mQvalueImage);
  mQvalueFilename = mSaveFilename;
  mQvalueImage.EnableSquaredImage(false);
  mQvalueImage.EnableUncertaintyImage(false);
  mQvalueImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mQvalueImage.Allocate();
  mQvalueImage.SetFilename(mQvalueFilename);
  mQvalueImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateQvalueActor::SaveData() {
  GateVActor::SaveData();
  mQvalueImage.SaveData(mCurrentEvent+1);
  G4cout<<std::endl;
  for (std::map<G4String,G4double>::iterator it=listOfEmiss.begin(); it!=listOfEmiss.end(); it++)
    G4cout<<(*it).first<<"  "<<(*it).second<<std::endl;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateQvalueActor::ResetData() {
  mCurrentEvent = -1;
  mQvalueImage.Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateQvalueActor::UserSteppingActionInVoxel(const int index, const G4Step* step)
{
  G4TrackVector* trackVector = const_cast<G4TrackVector*> (step->GetSecondary());
  int k = 0;

  double energyKinPre =  step->GetPreStepPoint()->GetKineticEnergy();
  double energyKinPost = step->GetPostStepPoint()->GetKineticEnergy();
  double energyKinSec = 0.;
  // int nSec = step->GetSecondary()->size(); // unused now (keep it for debug)

  k = 0;

  for(std::vector<G4Track*>::iterator it = trackVector->begin(); it != trackVector->end();it++ )
    {

      //G4cout<<"Second no = "<<k<<"   "<<(*it)->GetDefinition()->GetParticleName()<<"   "<<mNSec ;
      //if(k>=mNSec) G4cout<<"  OK    ";
      //else G4cout<<G4endl;
      if(k>=mNSec){
	energyKinSec+= (*it)->GetKineticEnergy();
	//G4cout<< (*it)->GetKineticEnergy()  <<G4endl;

      }
      k++;
    }

  double en = energyKinPre-energyKinPost-energyKinSec-step->GetTotalEnergyDeposit();

  mQvalueImage.AddValue(index, en);

  if(step->GetPostStepPoint()->GetProcessDefinedStep()){
    if(listOfEmiss[step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()]) listOfEmiss[step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()]+=en;
    else listOfEmiss[step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()]=en;
  }
  k = 0;

  for(std::vector<G4Track*>::iterator it = trackVector->begin(); it != trackVector->end();it++ )
    {

      if(k>=mNSec) mNSec++;
      k++;

    }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateQvalueActor::BeginOfRunAction(const G4Run * ) {
  GateDebugMessage("Actor", 3, "GateQvalueActor -- Begin of Run" << G4endl);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateQvalueActor::BeginOfEventAction(const G4Event * ) {
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateQvalueActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateQvalueActor::EndOfEventAction(const G4Event * ) {

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateQvalueActor::UserPreTrackActionInVoxel(const  int /*index*/, const G4Track* /*t*/)
{
  mNSec = 0;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateQvalueActor::UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/)
{
  //listOfPositionAndEnergy3[t->GetTrackID()] = listOfPositionAndEnergy2;
}
//-----------------------------------------------------------------------------
