/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \class  GateVoxelizedMassActor
  \author Thomas DESCHLER (thomas.deschler@iphc.cnrs.fr)
  \date	October 2015
*/

/*
  \brief Class GateVoxelizedMassActor :
  \brief
*/

#include "GateVoxelizedMassActor.hh"
#include "GateMiscFunctions.hh"

// g4
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateVoxelizedMassActor::GateVoxelizedMassActor(G4String name, G4int depth)
  :GateVImageActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateVoxelizedMassActor() -- begin\n");

  mIsMassImageEnabled = true;

  pMessenger = new GateVoxelizedMassActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateVoxelizedMassActor() -- end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateVoxelizedMassActor::~GateVoxelizedMassActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMassActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateVoxelizedMassActor -- Construct - begin\n");
  GateVImageActor::Construct();

  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(false);

  // Check if at least one image is enabled
  if (!mIsMassImageEnabled)
  {
    GateError("The VoxelizedMassActor " << GetObjectName()
              << " does not have any image enabled ...\n Please select at least one ('enableMass true' for example)");
  }

  // Output Filename
  mMassFilename = G4String(removeExtension(mSaveFilename))+"-Mass."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mMassImage);

  if (mIsMassImageEnabled) {
    mMassImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mMassImage.Allocate();
  }

  // Print information
  GateMessage("Actor", 1,
              "\tVoxelizedMassActor    = '" << GetObjectName() << "'\n" <<
              "\tMassFilename  = " << mMassFilename << Gateendl);

  ResetData();
  GateMessageDec("Actor", 4, "GateVoxelizedMassActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMassActor::SaveData()
{
  GateVActor::SaveData(); 

  if (mIsMassImageEnabled)
  {
    mMassImage.Write(mMassFilename);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMassActor::ResetData()
{
  if (mIsMassImageEnabled) mMassImage.Fill(0);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMassActor::BeginOfRunAction(const G4Run * r)
{
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateVoxelizedMassActor -- Begin of Run\n");

  if(mIsMassImageEnabled)
  {
    pVoxelizedMass.Initialize(mVolumeName,mMassImage);
    voxelMass.clear();
    voxelMass=pVoxelizedMass.GetVoxelMassVector();

    for(size_t i=0;i<voxelMass.size();i++)
      mMassImage.AddValue(i,voxelMass[i]/kg);
  }
}
//-----------------------------------------------------------------------------
