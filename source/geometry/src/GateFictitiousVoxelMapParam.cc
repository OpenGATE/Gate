/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateFictitiousVoxelMapParam.hh"
#include "GateRegularParameterization.hh"
#include "GateFictitiousVoxelMapParameterized.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "G4PVParameterised.hh"


///////////////////
//  Constructor  //
///////////////////

GateFictitiousVoxelMapParam::GateFictitiousVoxelMapParam(const  G4String& itsName, GateFictitiousVoxelMapParameterized* rpi):
   GateBox(itsName,"Vacuum",1,1,1,false,false),
    itsInserter(rpi),m_parameterization(0),m_pvParameterized(0)
{
}



//////////////////
//  Destructor  //
//////////////////

GateFictitiousVoxelMapParam::~GateFictitiousVoxelMapParam()
{
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "+-+- Entering GateFictitiousVoxelMapParam::Destructor ..."
           << G4endl << std::flush;
  }
}

//////////////////////////////////
//  ConstructOwnPhysicalVolume  //
//////////////////////////////////

void GateFictitiousVoxelMapParam::ConstructOwnPhysicalVolume(G4bool flagUpdate)
{
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "++++ Entering GateFictitiousVoxelMapParam::ConstructOwnPhysicalVolume ..."
           << G4endl << std::flush;
  }

  // For the update case; there is nothing to do here.
  if (flagUpdate) {
    if (itsInserter->GetVerbosity()>=1) {
      G4cout << "---- Exiting GateFictitiousVoxelMapParam::ConstructOwnPhysicalVolume ..."
             << G4endl
             << "     --> with flagUpdate = true"
             << G4endl << std::flush;
    }
    return;
  }

//  DestroyOwnPhysicalVolume();

  // Build the parameterization
  GateVGeometryVoxelReader* itsReader ( itsInserter->GetReader() );
  G4ThreeVector voxelNumber(itsReader->GetVoxelNx(),itsReader->GetVoxelNy(),itsReader->GetVoxelNz());
  m_parameterization = new GateRegularParameterization(itsInserter,voxelNumber);
  m_parameterization->BuildRegularParameterization();

  // Build the physical volume
  m_pvParameterized = new G4PVParameterised( mName+"_PVP",
                                             GetCreator()->GetLogicalVolume(),
                                             itsInserter->GetCreator()->GetLogicalVolume(),
                                             kXAxis,
                                             m_parameterization->GetNbOfCopies(),
                                             m_parameterization );

  //itsInserter->GetCreator()->GetLogicalVolume()->SetSmartless(0.02);
  // Set this physical volume as having a regular structure of type 1
  m_pvParameterized->SetRegularStructureId(1);
  // And finally push it into the physical volumes vector
  PushPhysicalVolume(m_pvParameterized);

  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "---- Exiting GateFictitiousVoxelMapParam::ConstructOwnPhysicalVolume ..."
           << G4endl << std::flush;
  }
}

///////////////////////
//  DestroyGeometry  //
///////////////////////

void GateFictitiousVoxelMapParam::DestroyGeometry()
{
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "++++ Entering GateFictitiousVoxelMapParam::DestroyGeometry ..."
           << G4endl << std::flush;
  }

  if (m_parameterization)
    delete m_parameterization;

  m_parameterization=0;
  m_pvParameterized=0;
  GateVVolume::DestroyGeometry();
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "---- Exiting GateFictitiousVoxelMapParam::DestroyGeometry ..."
           << G4endl << std::flush;
  }
}
