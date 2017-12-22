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


  // need m_energyWindowNb
  // need m_inputDataChannelIDList
  int digi_id = 2;

  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  DD(fDM->GetModuleCapacity());
  DD(fDM->GetCollectionCapacity());

  const GateSingleDigiCollection * SDC;
  SDC = dynamic_cast<const GateSingleDigiCollection*>(fDM->GetDigiCollection(digi_id));
  if (!SDC) {
    DD("no digi collection");
    return; // nothing stored
  }
  DD(const_cast<GateSingleDigiCollection*>(SDC)->GetName());
  G4int n_digi = SDC->entries();
  DD(n_digi);
  for (G4int i = 0; i < n_digi; i++) { // nb head ()
    DD(i);
    // G4int headID = 0;//m_system->GetMainComponentID((*SDC)[i]->GetPulse());
    G4double xProj = (*SDC)[i]->GetLocalPos()[0]; // X FIXME
    G4double yProj = (*SDC)[i]->GetLocalPos()[2]; // Z FIXME
    DD(xProj);
    DD(yProj);
    mCurrentData.u = xProj;
    mCurrentData.v = yProj;
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
  DD(p);

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
