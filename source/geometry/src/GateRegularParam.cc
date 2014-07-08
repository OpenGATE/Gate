/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateRegularParam.hh"
#include "GateRegularParameterization.hh"
#include "GateRegularParameterized.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "G4PVParameterised.hh"
#include "GateBox.hh"
 
///////////////////
//  Constructor  //
///////////////////

GateRegularParam::GateRegularParam(const  G4String& itsName, GateRegularParameterized* rpi)
  : GateBox(itsName,"Vacuum",1,1,1,false,false),
    itsInserter(rpi),m_parameterization(0),m_pvParameterized(0)
{
}

//////////////////
//  Destructor  //
//////////////////

GateRegularParam::~GateRegularParam()
{
}

///////////////////////////////////
//  ConstructOwnPhysicalVolumes  //
///////////////////////////////////

void GateRegularParam::ConstructOwnPhysicalVolume(G4bool flagUpdate)
{
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "---- Exiting GateRegularParam::ConstructOwnPhysicalVolumes ..."
             << G4endl
             << "     --> without flagUpdate"        
	     << G4endl << std::flush;
  }

  // For the update case; there is nothing to do here.
  if (flagUpdate) {
    if (itsInserter->GetVerbosity()>=1) {
      G4cout << "---- Exiting GateRegularParam::ConstructOwnPhysicalVolumes ..."
             << G4endl
             << "     --> with flagUpdate = true"
             << G4endl << std::flush;
    }
    return;
  }

//  DestroyOwnPhysicalVolumes();
   //DestroyGeometry();
   
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

  // Set this physical volume as having a regular structure of type 1
  m_pvParameterized->SetRegularStructureId(1);
  // And finally push it into the physical volumes vector
  PushPhysicalVolume(m_pvParameterized);

  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "---- Exiting GateRegularParam::ConstructOwnPhysicalVolumes ..."
           << G4endl << std::flush;
  }
}

/////////////////////////////////
//  DestroyOwnPhysicalVolumes  //
/////////////////////////////////
void GateRegularParam::DestroyGeometry()
{
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "++++ Entering GateRegularParam::DestroyOwnPhysicalVolumes ..."
           << G4endl << std::flush;
  }

  if (m_parameterization)
    delete m_parameterization;

  m_parameterization=0;
  m_pvParameterized=0;
  GateVVolume::DestroyGeometry();
  if (itsInserter->GetVerbosity()>=1) {
    G4cout << "---- Exiting GateRegularParam::DestroyOwnPhysicalVolumes ..."
           << G4endl << std::flush;
  }
}
