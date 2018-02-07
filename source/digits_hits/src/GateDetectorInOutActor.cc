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
void DetectorInData::Print(std::ostream & os)
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
void DetectorOutData::Print(std::ostream & os)
{
  os << "out = "
    // << u << " "
    // << v << " "
     << w << std::endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
GateDetectorInOutActor::GateDetectorInOutActor(G4String name, G4int depth) :
  GateVActor(name, depth)
{
  GateDebugMessageInc("Actor",4,"GateDetectorInOutActor() -- begin\n");
  pMessenger = new GateDetectorInOutActorMessenger(this);
  mOutputInDataOnlyFlag = false;
  mMaxAngle = 0.0; // no max angle
  mRRFactor = 0;   // no Russian Roulette factor
  mThetaMax = 0.0;
  mPhiMax = 0.0;
  GateDebugMessageDec("Actor",4,"GateDetectorInOutActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDetectorInOutActor::~GateDetectorInOutActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetOutputWindowNames(std::string & names)
{
  std::vector<std::string> words;
  GetWords(words, names);
  for(auto w:words) mListOfWindowNames.push_back(w);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetOutputInDataOnlyFlag(bool b)
{
  mOutputInDataOnlyFlag = b;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetMaxAngle(double a)
{
  mMaxAngle = a/deg;
  DD(mMaxAngle);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::SetRRFactor(int f)
{
  mRRFactor = f;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateDetectorInOutActor -- Construct - begin\n");
  GateVActor::Construct();

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
  GateMessage("Actor", 1, "GateDetectorInOutActor -> Detected " << mNumberOfDetectedEvent
              << " / " << mInData.size()
              << " = " << (double)mNumberOfDetectedEvent/(double)mInData.size()*100.0
              << "%" << std::endl);
  GateMessage("Actor", 1, "GateDetectorInOutActor -> max angles "
              << mThetaMax << " " << mPhiMax << std::endl);
  // Output simple binary file
  std::ofstream os;
  os.open(mSaveFilename, std::ios::out | std::ios::binary);
  auto s_in = sizeof(DetectorInData);
  auto s_out = sizeof(DetectorOutData);

  if (mOutputInDataOnlyFlag) {
    // Write In data only
    for(unsigned int i=0; i<mInData.size(); i++) {
      os.write(reinterpret_cast<char*>(&mInData[i]), s_in);
    }
  }
  else {
    // Write In and Out data
    for(unsigned int i=0; i<mInData.size(); i++) {
      os.write(reinterpret_cast<char*>(&mInData[i]), s_in);
      os.write(reinterpret_cast<char*>(&mOutData[i]), s_out);
    }
  }
  os.close();

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::ResetData()
{
  mInData.clear();
  mOutData.clear();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::BeginOfRunAction(const G4Run * r)
{
  GateVActor::BeginOfRunAction(r);
  mNumberOfDetectedEvent = 0;
  if (!mOutputInDataOnlyFlag) {
    G4DigiManager * fDM = G4DigiManager::GetDMpointer();
    for(auto name:mListOfWindowNames) {
      auto id = fDM->GetDigiCollectionID(name);
      GateMessage("Actor", 1, "GateDetectorInOutActor -> energy window " <<
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
void GateDetectorInOutActor::BeginOfEventAction(const G4Event * e)
{
  GateVActor::BeginOfEventAction(e);
  mEventIsAlreadyStored = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActor::EndOfEventAction(const G4Event * e)
{
  static int russian_roulette_current = 0;
  GateVActor::EndOfEventAction(e);
  if (!mOutputInDataOnlyFlag) {
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
      // mCurrentData.Print(std::cout);
    }
    if (!isIn) {
      if (mRRFactor == 0 or russian_roulette_current == mRRFactor-1) {
        mCurrentOutData.w = 0;               // windows 0 is 'outside'
        russian_roulette_current = 0;        // reset current value
        mOutData.push_back(mCurrentOutData); // store data
      }
      else { // ignore this event
        ++russian_roulette_current;
        mIgnoreCurrentData = true;
      }
    }
    else {
      mOutData.push_back(mCurrentOutData);
    }
  }
  // mCurrentData.Print(std::cout);
  // Do not count event that never go to UserSteppingAction
  double theta_center = 90.0;
  double phi_center = -90.0;
  if (mEventIsAlreadyStored and !mIgnoreCurrentData) {
    mInData.push_back(mCurrentInData);

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
void GateDetectorInOutActor::UserSteppingAction(const GateVVolume * /* v */, const G4Step* step)
{
  if (mEventIsAlreadyStored) return;

  // Get coordinate in the current volume coordinate system
  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable());
  int maxDepth = theTouchable->GetHistoryDepth();

  // Retrieve information
  //auto post = step->GetPostStepPoint();
  auto pre= step->GetPreStepPoint();
  auto p = theTouchable->GetHistory()->GetTransform(maxDepth).TransformPoint(pre->GetPosition());
  auto E = pre->GetKineticEnergy();
  auto dir = pre->GetMomentumDirection();
  // auto m = dir.mag(); // already == 1
  // auto ndir = dir/m;

  // which dimension ?? ask SPECThead system
  // https://en.wikipedia.org/wiki/Spherical_coordinate_system
  auto theta = acos(dir.z())/degree;
  auto phi = atan2(dir.y(),dir.x())/degree;

  // Threshold on angles: do not store if larger
  double theta_center = 90.0;
  double phi_center = -90.0;
  mIgnoreCurrentData = false;
  if (mOutputInDataOnlyFlag and // for OutputInDataOnly, check the angle max
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
  // mCurrentInData.Print(std::cout);

  // Output will be set EndOfEventAction
  mEventIsAlreadyStored = true;
}
//-----------------------------------------------------------------------------
