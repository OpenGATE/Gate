/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateDetectorInOutActor.hh"
#include "GateSingleDigi.hh"
#include "G4DigiManager.hh"

//-----------------------------------------------------------------------------
GateDetectorInOutActor::GateDetectorInOutActor(G4String name, G4int depth) :
  GateVActor(name, depth)
{
  DDF();
  GateDebugMessageInc("Actor",4,"GateDetectorInOutActor() -- begin\n");
  pMessenger = new GateDetectorInOutActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateDetectorInOutActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDetectorInOutActor::~GateDetectorInOutActor()
{
  DDF();
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetInputPlaneName(std::string & name)
{
  DDF();
  mInputPlaneName = name;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetOutputSystemName(std::string & name)
{
  DDF();
  mOutputSystemName = name;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetOutputWindowNames(std::string & names)
{
  DDF();
  DD(names);
  std::vector<std::string> words;
  GetWords(words, names);
  for(auto w:words) mListOfWindowNames.push_back(w);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::Construct()
{
  DDF();
  GateDebugMessageInc("Actor", 4, "GateDetectorInOutActor -- Construct - begin\n");
  GateVActor::Construct();
  DD(mInputPlaneName);
  DD(mOutputSystemName);

  /* Enable callbacks */
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  ResetData();
  GateMessageDec("Actor", 4, "GateDetectorInOutActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SaveData()
{
  DD(mSaveFilename);

  // trial simple binary file (associated txt ?)
  DD(mData.size());
  std::ofstream os;
  os.open(mSaveFilename, std::ios::out | std::ios::binary);
  os.write(reinterpret_cast<char*>(&mData[0]), mData.size()*sizeof(DetectorInOutData));
  os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::ResetData()
{
  DDF();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::BeginOfRunAction(const G4Run * r)
{
  GateVActor::BeginOfRunAction(r);
  DDF();
  DD(mOutputSystemName);

 G4DigiManager * fDM = G4DigiManager::GetDMpointer();
 for(auto name:mListOfWindowNames) {
   DD(name);
   auto id = fDM->GetDigiCollectionID(name);
   DD(id);
   if (id == -1) {
     GateError("Cannot find the energy window named: " << name);
   }
   mListOfWindowIds.push_back(id);
 }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
  mEventIsAlreadyStored = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::EndOfEventAction(const G4Event * e)
{
  GateVActor::EndOfEventAction(e);
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  bool isIn = false;
  int i=-1;
  for(auto id:mListOfWindowIds) {
    ++i;
    auto SDC = dynamic_cast<const GateSingleDigiCollection*>(fDM->GetDigiCollection(id));
    if (!SDC) continue;
    G4double xProj = (*SDC)[0]->GetLocalPos()[0]; // X FIXME ?
    G4double yProj = (*SDC)[0]->GetLocalPos()[1]; // Z FIXME ?
    mCurrentData.u = xProj;
    mCurrentData.v = yProj;
    mCurrentData.w = i;
    isIn = true;
  }
  if (!isIn) {
    mCurrentData.u = 0;
    mCurrentData.v = 0;
    mCurrentData.w = i;
  }
  mData.push_back(mCurrentData);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::UserSteppingAction(const GateVVolume * /* v */, const G4Step* step)
{
  if (mEventIsAlreadyStored) return;

  // Retrieve information
  auto track = step->GetTrack();
  auto p = track->GetPosition();
  auto E = track->GetKineticEnergy();
  auto dir = track->GetMomentumDirection();///E;
  auto l = dir.mag();
  // which dimension ?? ask SPECThead system
  auto theta = acos(dir.x()/l)/degree;
  auto phi = acos(dir.z()/l)/degree;

  // Input
  mCurrentData.x = p.x();
  mCurrentData.y = p.z();
  mCurrentData.E = E;
  mCurrentData.theta = theta;
  mCurrentData.phi = phi;

  // Output will be set EndOfEventAction
  mEventIsAlreadyStored = true;
}
//-----------------------------------------------------------------------------
