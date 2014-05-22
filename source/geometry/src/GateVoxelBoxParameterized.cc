/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVoxelBoxParameterized.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "GateGeometryVoxelInterfileReader.hh"

#include "GateOutputMgr.hh"
#include "GateVVolume.hh"
#include "GateBox.hh"
#include "GateVoxelOutput.hh"
#include "GateVOutputModule.hh"
#include "G4VisAttributes.hh"
#include "G4PVPlacement.hh"
#include "GatePlacementQueue.hh"
 
void GateVoxelBoxParameterized::InsertReader(G4String readerType ){
  //G4cout << "GateVoxelBoxParameterized::InsertReader - Entered" << G4endl;
  
  if (m_voxelReader) {
    G4cout << "GateVoxelBoxParameterized::InsertReader: voxel reader already defined" << G4endl;
    return;
  }

  if (readerType == G4String("image")){
    m_voxelReader = new GateGeometryVoxelImageReader(this);
  } else if (readerType == G4String("interfile")) {
    m_voxelReader = new GateGeometryVoxelInterfileReader(this);
  } else
    G4cout << "GateVoxelBoxParameterized::InsertReader: unknown reader type" << G4endl;
	
   // initialize voxel sizes with fake values  

       m_voxelReader->SetVoxelNx(1);
       m_voxelReader->SetVoxelNy(1);
       m_voxelReader->SetVoxelNz(1);  

  
}

void GateVoxelBoxParameterized::RemoveReader(){
  //  G4cout << "GateVoxelBoxParameterized::RemoveReader - Entered" << G4endl;
}

void GateVoxelBoxParameterized::AttachPhantomSD(){
  //  G4cout << "GateVoxelBoxParameterized::AttachPhantomSD - Entered for " << m_name << G4endl;
  m_voxelInserter->GetCreator()->AttachPhantomSD();
}

void GateVoxelBoxParameterized::AddOutput(G4String name){
  GateOutputMgr*     mgr ( GateOutputMgr::GetInstance() );
  mgr->AddOutputModule( (GateVOutputModule*) new GateVoxelOutput(name, GetObjectName(), mgr, mgr->GetDigiMode(), this)  );
}


void GateVoxelBoxParameterized::ConstructGeometry(G4LogicalVolume* mother_log, G4bool flagUpdateOnly){
  // G4cout << "GateVoxelBoxParameterized::ConstructGeometry - Entered for " << GetCreator()->GetObjectName() 
  //  	 <<", flag "<< std::boolalpha << flagUpdateOnly << G4endl;
    
  if (m_voxelReader){
    //  Get the voxel number and size from the reader
    voxelNumber = G4ThreeVector(m_voxelReader->GetVoxelNx(), m_voxelReader->GetVoxelNy(), m_voxelReader->GetVoxelNz());
    voxelSize   = G4ThreeVector(m_voxelReader->GetVoxelSize());
  }else{
    G4cout << "GateVoxelBoxParameterized::ConstructGeometry - Warning ! ConstructGeometry called without a reader" << G4endl;
    return;
  }
    
  //  G4cout << "GateVoxelBoxParameterized::ConstructGeometry - voxel number/size " << voxelNumber << " . " << voxelSize << G4endl << std::flush;
  GateBox* m_boxCreator;

  //  Update the dimensions of the enclosing box
  m_boxCreator = dynamic_cast<GateBox*>( GetCreator() );
  m_boxCreator->SetBoxXLength( voxelNumber.x() * voxelSize.x() );
  m_boxCreator->SetBoxYLength( voxelNumber.y() * voxelSize.y());
  m_boxCreator->SetBoxZLength( voxelNumber.z() * voxelSize.z());

  //  Update the dimensions of the voxel box
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

