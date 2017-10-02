/*----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateParticleFilter.hh"
#include "GateUserActions.hh"
#include "GateTrajectory.hh"

//---------------------------------------------------------------------------
GateParticleFilter::GateParticleFilter(G4String name)
  : GateVFilter(name)
{
  thePdef.clear();
  pPartMessenger = new GateParticleFilterMessenger(this);
  nFilteredParticles = 0;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
GateParticleFilter::~GateParticleFilter()
{
  if (nFilteredParticles == 0) GateWarning("No particle has been selected by filter: " << GetObjectName());
  delete pPartMessenger ;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateParticleFilter::Accept(const G4Track *aTrack)
{
//<<<<<<< HEAD
  //G4bool accept = true;

  //// Test the particle type
  //if (!thePdef.empty()) {
    //accept = false;
//=======
  //G4bool accept = true;
  std::vector<bool> acceptTemp;// = {true};
  acceptTemp.push_back(true);
  // Test the particle type
  if (!thePdef.empty()) {
    acceptTemp.push_back(false);
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
    for (size_t i = 0; i < thePdef.size(); i++) {
      if (thePdef[i] == aTrack->GetDefinition()->GetParticleName() ||
          (aTrack->GetDefinition()->GetParticleSubType() == "generic" && thePdef[i] == "GenericIon") ) {
        nFilteredParticles++;
//<<<<<<< HEAD
        //accept = true;
//=======
        acceptTemp.back()=true;
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
        break;
      }
    }
  } // end thePdef !empty
//<<<<<<< HEAD
  //if (!accept) return false;

  //// Test the particle Z
  //if (!thePdefZ.empty()) {
    //accept = false;
    //for (size_t i = 0; i < thePdefZ.size(); i++) {
      //if (thePdefZ[i] == aTrack->GetDefinition()->GetAtomicNumber()) {
        //nFilteredParticles++;
        //accept = true;
//=======
  //if (!accept) return false;

  // Test the particle Z
  if (!thePdefZ.empty()) {
    //accept = false;
    acceptTemp.push_back(false);
    //acceptTemp.push_back(false);
    for (size_t i = 0; i < thePdefZ.size(); i++) {
      if (thePdefZ[i] == aTrack->GetDefinition()->GetAtomicNumber()) {
        nFilteredParticles++;
        //accept = true;
        acceptTemp.back()=true;
        //acceptTemp[-1]=true;
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
        break;
      }
    }
  } // end thePdefZ !empty
//<<<<<<< HEAD
  //if (!accept) return false;

  //// Test the parent
  //if (!theParentPdef.empty()) {
    //accept = false;
//=======
  // Test the particle A
  //G4cout<<"empty A?: "<<thePdefA.empty()<<G4endl;
  if (!thePdefA.empty()) {
      //G4cout<<"length: "<<acceptTemp.size()<<G4endl;
    acceptTemp.push_back(false);
      //G4cout<<"length: "<<acceptTemp.size()<<G4endl;
    for (size_t i = 0; i < thePdefA.size(); i++) {
      if (thePdefA[i] == aTrack->GetDefinition()->GetAtomicMass()) {
        nFilteredParticles++;
        //accept = true;
        acceptTemp.back()=true;
       //G4cout<<"length: "<<acceptTemp.size()<<G4endl;
        break;
      }
    }
  } // end thePdefA !empty
  //if (accept==true){
  //G4cout<<"acceptTemp: "<<" "<<acceptTemp[0]<<" "<<acceptTemp[1]<<" "<<acceptTemp[2]<<G4endl;
  //}
  
            
  //if (!accept)
  //{
      ////G4cout<<"ai ai ai"<<G4endl;
      //return false;
  //}
  // Test the particle PDG
  if (!thePdefPDG.empty()) {
    acceptTemp.push_back(false);
    for (size_t i = 0; i < thePdefPDG.size(); i++) {
      if (thePdefPDG[i] == aTrack->GetDefinition()->GetPDGEncoding()) {
        nFilteredParticles++;
        acceptTemp.back()=true;
        //G4cout<<"PDG: "<<aTrack->GetDefinition()->GetPDGEncoding()<<G4endl;
        break;
      }
    }
  } // end thePdefPFG !empty
  //if (!accept) return false;

  // Test the parent
  if (!theParentPdef.empty()) {
    acceptTemp.push_back(false);
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
    GateTrackIDInfo * trackInfo =
      GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    while (trackInfo) {
      for (size_t i = 0; i < theParentPdef.size(); i++) {
        if (theParentPdef[i] == trackInfo->GetParticleName()) {
          nFilteredParticles++;
          acceptTemp.back()=true;
          break;
        }
      }
//<<<<<<< HEAD
      //if (accept == true) break;
//=======
      if (acceptTemp.back() == true) break;
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
      int id = trackInfo->GetParentID();
      trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(id);
    }
  } // end theParentPdef !empty
//<<<<<<< HEAD
  //if (!accept) return false;
//=======
  //if (!accept) return false;
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630


  // Test the directParent
  if (!theDirectParentPdef.empty()) {
//<<<<<<< HEAD
    //accept = false;
//=======
    acceptTemp.push_back(false);
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
    GateTrackIDInfo * trackInfo =
      GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    if (trackInfo) {
      for (size_t i = 0; i < theDirectParentPdef.size(); i++) {
        if (theDirectParentPdef[i] == trackInfo->GetParticleName()) {
          nFilteredParticles++;
          acceptTemp.back()=true;
          break;
        }
      }
    }
  } // end theDirectParentPdef !empty

//<<<<<<< HEAD
  //return accept;
//=======
   if (!(std::all_of(std::begin(acceptTemp), std::end(acceptTemp),[](bool i){return i == true;})))
            {
                //G4cout<<"Z: "<< aTrack->GetDefinition()->GetAtomicNumber()<<G4endl;
                //G4cout<<"A: "<< aTrack->GetDefinition()->GetAtomicMass()<<G4endl;
                 //G4cout<<"Accept: "<<" "<<acceptTemp[0]<<" "<<acceptTemp[1]<<" "<<acceptTemp[2]<<G4endl;
                return false;}
   else return true;
  //return accept;
//>>>>>>> a192603ade15370f2eb1a20558e3de03d1898630
}

//---------------------------------------------------------------------------
void GateParticleFilter::Add(const G4String &particleName)
{
  for (size_t i = 0; i < thePdef.size(); i++) {
    if (thePdef[i] == particleName ) return;
  }
  thePdef.push_back(particleName);
}

//---------------------------------------------------------------------------
void GateParticleFilter::AddZ(const G4int &particleZ)
{
  for (size_t i = 0; i < thePdefZ.size(); i++) {
    if (thePdefZ[i] == particleZ ) return;
  }
  thePdefZ.push_back(particleZ);
}
//---------------------------------------------------------------------------
void GateParticleFilter::AddA(const G4int &particleA)
{
  for (size_t i = 0; i < thePdefA.size(); i++) {
    if (thePdefA[i] == particleA ) return;
  }
  thePdefA.push_back(particleA);
}
//---------------------------------------------------------------------------
void GateParticleFilter::AddPDG(const G4int &particlePDG)
{
  for (size_t i = 0; i < thePdefPDG.size(); i++) {
    if (thePdefPDG[i] == particlePDG ) return;
  }
  thePdefPDG.push_back(particlePDG);
}
//---------------------------------------------------------------------------

void GateParticleFilter::AddParent(const G4String &particleName)
{
  for (size_t i = 0; i < theParentPdef.size(); i++) {
    if (theParentPdef[i] == particleName ) return;
  }
  theParentPdef.push_back(particleName);
}
//---------------------------------------------------------------------------

void GateParticleFilter::AddDirectParent(const G4String &particleName)
{
  for (size_t i = 0; i < theDirectParentPdef.size(); i++) {
    if (theDirectParentPdef[i] == particleName ) return;
  }
  theDirectParentPdef.push_back(particleName);
}

//---------------------------------------------------------------------------
void GateParticleFilter::show() {
  G4cout << "------ Filter: " << GetObjectName() << " ------" << G4endl;
  G4cout << "     particle list:" << G4endl;

  for (size_t i = 0; i < thePdef.size(); i++) {
    G4cout << thePdef[i] << G4endl;
  }
  G4cout << "     parent particle list:" << G4endl;
  for (size_t i = 0; i < theParentPdef.size(); i++) {
    G4cout << theParentPdef[i] << G4endl;
  }
  G4cout << "-------------------------------------------" << G4endl;
}
//---------------------------------------------------------------------------
