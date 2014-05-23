/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateCompressedVoxelParameterized.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "GateGeometryVoxelInterfileReader.hh"
#include "GateOutputMgr.hh"
#include "GateBox.hh"
#include "GateCompressedVoxelOutput.hh"
#include "GateVOutputModule.hh"
#include "G4VisAttributes.hh"
#include "G4PVPlacement.hh"
#include "GatePlacementQueue.hh"

//---------------------------------------------------------------------------
void GateCompressedVoxelParameterized::InsertReader(G4String readerType){
  
  if (m_voxelReader) {
    return;
  }

  if (readerType == G4String("image")){
    m_voxelReader = new GateGeometryVoxelImageReader(this);
    m_voxelReader->CreateCompressor();

  } else if (readerType == G4String("interfile")) {
    m_voxelReader = new GateGeometryVoxelInterfileReader(this);
    m_voxelReader->CreateCompressor();
  } else

  // initialize voxel sizes with fake values  

       m_voxelReader->SetVoxelNx(10);
       m_voxelReader->SetVoxelNy(10);
       m_voxelReader->SetVoxelNz(10);

}
//---------------------------------------------------------------------------


void GateCompressedVoxelParameterized::RemoveReader(){
}
//---------------------------------------------------------------------------


void GateCompressedVoxelParameterized::AttachPhantomSD(){
   m_voxelInserter->GetCreator()->AttachPhantomSD();

}
//---------------------------------------------------------------------------


void GateCompressedVoxelParameterized::AddOutput(G4String name){
  GateOutputMgr*     mgr ( GateOutputMgr::GetInstance() );
  mgr->AddOutputModule( (GateVOutputModule*) new GateCompressedVoxelOutput(name, GetObjectName(), mgr, mgr->GetDigiMode(), this)  );
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCompressedVoxelParameterized::ConstructGeometry(G4LogicalVolume* mother_log, G4bool flagUpdateOnly){
   
  if (m_voxelReader){
    //  Get the original number of voxels and size from the reader
    voxelNumber = G4ThreeVector(m_voxelReader->GetVoxelNx(), m_voxelReader->GetVoxelNy(), m_voxelReader->GetVoxelNz());
    voxelSize   = G4ThreeVector(m_voxelReader->GetVoxelSize());
  }else{
    return;
  }
    
  GateBox* m_boxCreator;

  //  Update the dimensions of the enclosing box
  m_boxCreator = dynamic_cast<GateBox*>( GetCreator() );
  m_boxCreator->SetBoxXLength( voxelNumber.x() * voxelSize.x() );
  m_boxCreator->SetBoxYLength( voxelNumber.y() * voxelSize.y());
  m_boxCreator->SetBoxZLength( voxelNumber.z() * voxelSize.z());

  // Update the dimensions of the voxel box

  m_boxCreator = dynamic_cast<GateBox*>(m_voxelInserter->GetCreator());
  m_boxCreator->SetBoxXLength( voxelSize.x() );
  m_boxCreator->SetBoxYLength( voxelSize.y() );
  m_boxCreator->SetBoxZLength( voxelSize.z() );


  //  Proceed with the rest
  GateVVolume::ConstructGeometry(mother_log, flagUpdateOnly);



  //  Visibility attributes
  G4VisAttributes* creatorVis= const_cast<G4VisAttributes*>(GetCreator()->GetLogicalVolume()->GetVisAttributes());
  creatorVis->SetForceWireframe(true);
}
//---------------------------------------------------------------------------

