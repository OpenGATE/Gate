/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "Gate_NN_ARF_Actor.hh"
#include "GateSingleDigi.hh"
#include "G4DigiManager.hh"

//-----------------------------------------------------------------------------
void Gate_NN_ARF_Input_Data::Print(std::ostream & os)
{
  os << " in = "
     << x << " "
     << y << " "
     << theta << " "
     << phi << " "
     << E << " "
     << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void Gate_NN_ARF_Output_Data::Print(std::ostream & os)
{
  os << "out = " << w << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::Gate_NN_ARF_Actor(G4String name, G4int depth) :
  GateVActor(name, depth)
{
  GateDebugMessageInc("Actor",4,"Gate_NN_ARF_Actor() -- begin\n");
  pMessenger = new Gate_NN_ARF_ActorMessenger(this);
  mTrainingModeFlag = false;
  mMaxAngle = 0.0; // no max angle
  mRRFactor = 0;   // no Russian Roulette factor
  mThetaMax = 0.0;
  mPhiMax = 0.0;
  GateDebugMessageDec("Actor",4,"Gate_NN_ARF_Actor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Gate_NN_ARF_Actor::~Gate_NN_ARF_Actor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetEnergyWindowNames(std::string & names)
{
  std::vector<std::string> words;
  GetWords(words, names);
  for(auto w:words) mListOfWindowNames.push_back(w);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetMode(std::string m)
{
  DD(m);
  if (m == "train") mTrainingModeFlag = true;
  else {
    if (m == "test") mTrainingModeFlag = false;
    else {
      GateError("Error in Gate_NN_ARF_Actor macro 'setMode', must be 'train' or 'test', while read " << m);
    }
  }
  DD(mTrainingModeFlag);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetMaxAngle(double a)
{
  mMaxAngle = a/deg;
  DD(mMaxAngle);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SetRRFactor(int f)
{
  mRRFactor = f;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::Construct()
{
  GateDebugMessageInc("Actor", 4, "Gate_NN_ARF_Actor -- Construct - begin\n");
  GateVActor::Construct();

  /* Enable callbacks */
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  ResetData();
  GateMessageDec("Actor", 4, "Gate_NN_ARF_Actor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::SaveData()
{
  GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> Detected "
              << mNumberOfDetectedEvent
              << " / " << mInData.size()
              << " = " << (double)mNumberOfDetectedEvent/(double)mInData.size()*100.0
              << "%" << std::endl);
  GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> max angles "
              << mThetaMax << " " << mPhiMax << std::endl);

  // Output simple binary file
  std::ofstream os;
  os.open(mSaveFilename, std::ios::out | std::ios::binary);
  auto s_in = sizeof(Gate_NN_ARF_Input_Data);
  auto s_out = sizeof(Gate_NN_ARF_Output_Data);

  if (mTrainingModeFlag) {
    // Write In and Out data
    DD(mInData.size());
    DD(mOutData.size());
    for(unsigned int i=0; i<mInData.size(); i++) {
      os.write(reinterpret_cast<char*>(&mInData[i]), s_in);
      os.write(reinterpret_cast<char*>(&mOutData[i]), s_out);
    }
  }
  else {
    // Write In data only
    for(unsigned int i=0; i<mInData.size(); i++) {
      os.write(reinterpret_cast<char*>(&mInData[i]), s_in);
    }
  }
  os.close();

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::ResetData()
{
  mInData.clear();
  mOutData.clear();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::BeginOfRunAction(const G4Run * r)
{
  GateVActor::BeginOfRunAction(r);
  mNumberOfDetectedEvent = 0;
  if (mTrainingModeFlag) {
    G4DigiManager * fDM = G4DigiManager::GetDMpointer();
    for(auto name:mListOfWindowNames) {
      auto id = fDM->GetDigiCollectionID(name);
      GateMessage("Actor", 1, "Gate_NN_ARF_Actor -> energy window " <<
                  name << " (id = " << id << ")" << std::endl);
      if (id == -1) {
        GateError("Cannot find the energy window named: " << name);
      }
      mListOfWindowIds.push_back(id);
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
  mEventIsAlreadyStored = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::EndOfEventAction(const G4Event * e)
{
  static int russian_roulette_current = 0;
  GateVActor::EndOfEventAction(e);
  if (mTrainingModeFlag) {
    G4DigiManager * fDM = G4DigiManager::GetDMpointer();
    bool isIn = false;
    int i=0;
    for(auto id:mListOfWindowIds) {
      ++i;
      auto SDC = dynamic_cast<const GateSingleDigiCollection*>(fDM->GetDigiCollection(id));
      if (!SDC) continue;
      /*
      // ==> No need for u,v coordinates
      G4double xProj = (*SDC)[0]->GetLocalPos()[0]; // X FIXME ?
      G4double yProj = (*SDC)[0]->GetLocalPos()[1]; // Z FIXME ?
      mCurrentOutData.u = xProj;
      mCurrentOutData.v = yProj;
      */
      mCurrentOutData.w = i;
      isIn = true;
      ++mNumberOfDetectedEvent;
    }
    if (!isIn) {
      if (mRRFactor == 0 or russian_roulette_current == mRRFactor-1) {
        mCurrentOutData.w = 0;               // windows 0 is 'outside'
        russian_roulette_current = 0;        // reset current value
        mOutData.push_back(mCurrentOutData); // store data
        //mCurrentOutData.Print(std::cout);
      }
      else { // ignore this event
        ++russian_roulette_current;
        mIgnoreCurrentData = true;
      }
    }
    else {
      mOutData.push_back(mCurrentOutData);
      //mCurrentOutData.Print(std::cout);
    }
  }

  // Do not count event that never go to UserSteppingAction
  double theta_center = 90.0;
  double phi_center = -90.0;
  if (mEventIsAlreadyStored and !mIgnoreCurrentData) {
    mInData.push_back(mCurrentInData);
    //mCurrentInData.Print(std::cout);
    // Check max angle
    if (mCurrentOutData.w != 0) { // not outside
      auto a = fabs(mCurrentInData.theta-theta_center);
      if (a > mThetaMax) {
        mThetaMax = a;
        DD(mThetaMax);
      }
      auto b = fabs(mCurrentInData.phi-phi_center);
      if (b > mPhiMax) {
        mPhiMax = b;
        DD(mPhiMax);
      }
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_Actor::UserSteppingAction(const GateVVolume * /* v */, const G4Step* step)
{
  if (mEventIsAlreadyStored) return;

  // Get coordinate in the current volume coordinate system
  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable());
  int maxDepth = theTouchable->GetHistoryDepth();

  // Get information
  auto pre= step->GetPreStepPoint();
  auto p = theTouchable->GetHistory()->GetTransform(maxDepth).TransformPoint(pre->GetPosition());
  auto E = pre->GetKineticEnergy();
  auto dir = pre->GetMomentumDirection();

  // which dimension ?? ask SPECThead system FIXME
  // https://en.wikipedia.org/wiki/Spherical_coordinate_system
  auto theta = acos(dir.z())/degree;
  auto phi = atan2(dir.y(),dir.x())/degree;

  // Threshold on angles: do not store if larger
  double theta_center = 90.0;
  double phi_center = -90.0;
  mIgnoreCurrentData = false;
  if (!mTrainingModeFlag
      and mMaxAngle != 0.0
      and // when test mode, check the angle max
      (theta > theta_center+mMaxAngle or
       theta < theta_center-mMaxAngle or
       phi   > phi_center+mMaxAngle or
       phi   < phi_center-mMaxAngle)) {
    mIgnoreCurrentData = true;
    mEventIsAlreadyStored = true;
    return;
  }

  // Input
  mCurrentInData.x = p.x();
  mCurrentInData.y = p.y();
  mCurrentInData.E = E;
  mCurrentInData.theta = theta;
  mCurrentInData.phi = phi;
  //mCurrentInData.Print(std::cout);

  // Output will be set EndOfEventAction
  mEventIsAlreadyStored = true;
}
//-----------------------------------------------------------------------------
