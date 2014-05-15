/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateProductionActor :
  \brief compute production count for filtered particles
*/

#ifndef GATEPRODUCTIONACTOR_CC
#define GATEPRODUCTIONACTOR_CC

#include "GateProductionActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateProductionActor::GateProductionActor(G4String name, G4int depth) :
  GateVImageActor(name,depth), pMessenger(NULL)
{
  GateMessage("Actor",2,"GateProductionActor -- constructor"<<G4endl);
  pMessenger = new GateProductionActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateProductionActor::~GateProductionActor()
{
  GateMessage("Actor",2,"GateProductionActor -- destructor"<<G4endl);
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateProductionActor::Construct()
{
  GateMessage("Actor",2,"GateProductionActor -- construct"<<G4endl);
  GateVImageActor::Construct();
  mImage.Allocate(); // allocate data and reset value to 0

  if (mSaveFilename.isNull() || mSaveFilename=="FilnameNotGivenForThisActor") { GateError("GateProductionActor -- please give output filename"); }
  if (!mVolume) { GateError("GateProductionActor -- please attach actor to a volume"); }

  GateMessage("Actor",3,"GateProductionActor -- filename=" << mSaveFilename <<G4endl);
  GateMessage("Actor",3,"GateProductionActor -- imagesize=" << mImage.GetNumberOfValues() <<G4endl);

  // Enable callbacks
  EnablePreUserTrackingAction(true);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Save production image
void GateProductionActor::SaveData()
{
  GateVActor::SaveData();
  GateMessage("Actor",2,"GateProductionActor -- saving filename="<< mSaveFilename <<G4endl);
  mImage.Write(mSaveFilename);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateProductionActor::ResetData() {
  mImage.Fill(0);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Start of track callback
void GateProductionActor::UserPreTrackActionInVoxel(const int index, const G4Track* /*track*/)
{
  if (index<0) return;
  assert(index>=0 && index<mImage.GetNumberOfValues());
  mImage.GetValue(index)++;
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEPRODUCTIONACTOR_CC */
