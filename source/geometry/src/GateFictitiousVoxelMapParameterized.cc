/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateFictitiousVoxelMapParameterized.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "GateGeometryVoxelInterfileReader.hh"
#include "GateOutputMgr.hh"
#include "GateBox.hh"
#include "GateVoxelOutput.hh"
#include "GateVOutputModule.hh"
#include "G4VisAttributes.hh"
#include "GatePlacementQueue.hh"


#include "G4Region.hh"
#include "GatePETVRTManager.hh"
#include "GatePETVRTSettings.hh"
#include "GateFictitiousFastSimulationModel.hh"
#include "GateFictitiousVoxelMapParam.hh"
typedef G4Region G4Envelope;
#include "GateFictitiousVoxelMapParameterizedMessenger.hh"
#include "G4ProductionCuts.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include <sstream>

#include "GateCylinder.hh"

///////////////////
//  Constructor  //
///////////////////

GateFictitiousVoxelMapParameterized::GateFictitiousVoxelMapParameterized ( const G4String& name,
						   				G4bool acceptsChildren, 
		 			   	   				G4int  depth) :
		GateBox ( name,"Vacuum",1,1,1,acceptsChildren,depth )
{
	Init ( name,GateFictitiousVoxelMapParameterized::Box );
}

GateFictitiousVoxelMapParameterized::GateFictitiousVoxelMapParameterized ( const G4String& name) :
		GateBox ( name,"Vacuum",1,1,1,false,false)
{
	Init ( name,GateFictitiousVoxelMapParameterized::Box );
}

void GateFictitiousVoxelMapParameterized::Init ( const G4String& name, GateFictitiousVoxelMapParameterized::EnvelopeType type )
{
	m_nEnvelopeType=type;
	m_name=name;
	m_messenger =new GateFictitiousVoxelMapParameterizedMessenger ( this );
	m_voxelReader=NULL;
	m_voxelInserter =new GateFictitiousVoxelMapParam ( name+"Voxel", this ) ;
	voxelNumber=G4ThreeVector ( 1,1,1 ) ;
	voxelSize =G4ThreeVector ( 1,1,1 ) ;
	verboseLevel=0; //! Default : set to quiet
  skipEqualMaterials = 0; //! Default: because G4 bug for speed up set at 1
        GetCreator()->GetTheChildList()->AddChild(m_voxelInserter);
	m_nGammaCut=-1;
	m_pProductionCuts=NULL;
	m_nEnvelopeType=type;
}

//////////////////
//  Destructor  //
//////////////////

GateFictitiousVoxelMapParameterized::~GateFictitiousVoxelMapParameterized()
{
	delete m_messenger;
	if ( m_pProductionCuts!=NULL )
		delete m_pProductionCuts;
}

////////////////////
//  InsertReader  //
////////////////////

void GateFictitiousVoxelMapParameterized::InsertReader ( G4String readerType )
{
	if ( verboseLevel>=1 )
	{
		G4cout << "++++ Entering GateFictitiousVoxelMapParameterized::InsertReader ..."
		<< G4endl << std::flush;
	}

	if ( m_voxelReader )
	{
		G4cout << "GateFictitiousVoxelMapParameterized::InsertReader: voxel reader already defined" << G4endl;
		return;
	}

	if ( readerType == G4String ( "image" ) )
	{
		m_voxelReader = new GateGeometryVoxelImageReader ( this );
	}
	else if ( readerType == G4String ( "interfile" ) )
	{
		m_voxelReader = new GateGeometryVoxelInterfileReader ( this );
	}
	else
		G4cout << "GateFictitiousVoxelMapParameterized::InsertReader: unknown reader type" << G4endl;

	if ( verboseLevel>=1 )
	{
		G4cout << "---- Exiting GateFictitiousVoxelMapParameterized::InsertReader ..."
		<< G4endl << std::flush;
	}
}

////////////////////
//  RemoveReader  //
////////////////////

void GateFictitiousVoxelMapParameterized::RemoveReader()
{
	if ( verboseLevel>=1 )
	{
		G4cout << "+-+- Entering GateFictitiousVoxelMapParameterized::RemoveReader ..."
		<< G4endl << std::flush;
	}
}

///////////////////////
//  AttachPhantomSD  //
///////////////////////

void GateFictitiousVoxelMapParameterized::AttachPhantomSD()
{
	if ( verboseLevel>=1 )
	{
		G4cout << "++++ Entering GateFictitiousVoxelMapParameterized::AttachPhantomSD ..."
		<< G4endl << std::flush;
	}

	m_voxelInserter->GetCreator()->AttachPhantomSD();

	GatePETVRTSettings* settings= GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings();
	settings->RegisterPhantomSD ( GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD(),false );


	if ( verboseLevel>=1 )
	{
		G4cout << "---- Exiting GateFictitiousVoxelMapParameterized::AttachPhantomSD ..."
		<< G4endl << std::flush;
	}
}

/////////////////
//  AddOutput  //
/////////////////

void GateFictitiousVoxelMapParameterized::AddOutput ( G4String name )
{
	if ( verboseLevel>=1 )
	{
		G4cout << "++++ Entering GateFictitiousVoxelMapParameterized::AddOutput ..."
		<< G4endl << std::flush;
	}

	GateOutputMgr* mgr ( GateOutputMgr::GetInstance() );
	mgr->AddOutputModule ( ( GateVOutputModule* ) new GateVoxelOutput ( name,GetObjectName(),mgr,mgr->GetDigiMode(),this ) );

	if ( verboseLevel>=1 )
	{
		G4cout << "---- Exiting GateFictitiousVoxelMapParameterized::AddOutput ..."
		<< G4endl << std::flush;
	}
}

/////////////////////////
//  ConstructGeometry  //
/////////////////////////

void GateFictitiousVoxelMapParameterized::ConstructGeometry ( G4LogicalVolume* mother_log, G4bool flagUpdateOnly )
{
	if ( verboseLevel>=1 )
	{
		G4cout << "++++ Entering GateFictitiousVoxelMapParameterized::ConstructGeometry ..."
		<< G4endl
		<< "     --> with : flagUpdateOnly = " << flagUpdateOnly
		<< "  |  mother_log = " << mother_log
		<< G4endl << std::flush;
	}

	if ( m_voxelReader )
	{
		// Get the voxel number and size from the reader
		voxelNumber = G4ThreeVector ( m_voxelReader->GetVoxelNx(),
		                              m_voxelReader->GetVoxelNy(),
		                              m_voxelReader->GetVoxelNz() );
		voxelSize   = G4ThreeVector ( m_voxelReader->GetVoxelSize() );
	}
	else
	{
		G4cout << "GateFictitiousVoxelMapParameterized::ConstructGeometry - Warning ! ConstructGeometry called without a reader" << G4endl << std::flush;
		return;
	}

	GateBox* m_boxCreator=NULL;
	// Update the dimensions of the enclosing box if G4Box
	switch ( m_nEnvelopeType )
	{
		case GateFictitiousVoxelMapParameterized::Box:
			m_boxCreator = dynamic_cast<GateBox*> ( GetCreator() );
			m_boxCreator->SetBoxXLength ( voxelNumber.x() * voxelSize.x() );
			m_boxCreator->SetBoxYLength ( voxelNumber.y() * voxelSize.y() );
			m_boxCreator->SetBoxZLength ( voxelNumber.z() * voxelSize.z() );
			break;
		default:
			exit ( EXIT_FAILURE );

	}

	// Update the dimensions of the voxel box
	m_boxCreator = dynamic_cast<GateBox*> ( m_voxelInserter->GetCreator() );
	m_boxCreator->SetBoxXLength ( voxelSize.x() );
	m_boxCreator->SetBoxYLength ( voxelSize.y() );
	m_boxCreator->SetBoxZLength ( voxelSize.z() );

	// Proceed with the rest
	GateVVolume::ConstructGeometry(mother_log, flagUpdateOnly);

	if ( m_nEnvelopeType!=GateFictitiousVoxelMapParameterized::Box )
	{
		if ( m_voxelInserter->GetParameterization()->SkipEqualMaterials() )
		{

			G4cout << "GateFictitiousVoxelMapParameterized::ConstructGeometry: SkipEqualMaterials of Parameterization set to 'false'. Value 'true' only possible for G4Box solid." << G4endl;
			m_voxelInserter->GetParameterization()->SetSkipEqualMaterials ( false );
		}
	}

	if ( !flagUpdateOnly )
	{
		G4Envelope* region=GetCreator()->GetLogicalVolume()->GetRegion();
		if ( region==NULL )
			G4Exception ( "GateFictitiousVoxelMapParameterized::ConstructGeometry", "ConstructGeometry", FatalException, "Cannot create/allocate G4Region! Aborting." );
		if ( m_pProductionCuts!=NULL )
			region->SetProductionCuts ( m_pProductionCuts );
		GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->RegisterEnvelope ( region );
		GateFictitiousVoxelMap* fmap=new GateFictitiousVoxelMap ( region );

		fmap->RegisterGeometryVoxelReader ( m_voxelReader,false );
		GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->RegisterFictitiousMap ( fmap,true ); // true==delete map together with settings
	}



	// Visibility attributes
	G4VisAttributes* creatorVis= const_cast<G4VisAttributes*> ( GetCreator()->GetLogicalVolume()->GetVisAttributes() );
	creatorVis->SetForceWireframe ( true );

	if ( verboseLevel>=1 )
	{
		G4cout << "---- Exiting GateFictitiousVoxelMapParameterized::ConstructGeometry ..."
		<< G4endl << std::flush;
	}
}

//////////////////////////////////
//  ConstructOwnPhysicalVolume  //
//////////////////////////////////

void GateFictitiousVoxelMapParameterized::ConstructOwnPhysicalVolume ( G4bool flagUpdateOnly )
{
  // Store the volume default position into a placement queue 
  GatePlacementQueue motherQueue;
  motherQueue.push_back(GatePlacement(G4RotationMatrix(),G4ThreeVector()));
    
  GatePlacementQueue *pQueue = &motherQueue;
        
  // Have the start-up position processed by the move list 
  if (m_moveList){
    pQueue = m_moveList->ComputePlacements(pQueue);
  }
  // Have the volume's current position processed by the repeater list
  if (m_repeaterList){
//    G4cout << " *** repeaterList exists, repeaterList->ComputePlacements(pQueue)" << G4endl;
    pQueue = m_repeaterList->ComputePlacements(pQueue);}

  
  // Do consistency checks
  if (flagUpdateOnly && theListOfOwnPhysVolume.size()) {
    if (pQueue->size()!=theListOfOwnPhysVolume.size()) {
      G4cout  << "[GateVVolume('" << GetObjectName() << "')::ConstructOwnPhysicalVolume]:" << G4endl
      	      << "The size of the placement queue (" << pQueue->size() << ") is different from " << G4endl 
	      << "the number of physical volumes to update (" << theListOfOwnPhysVolume.size() << ")!!!" << G4endl;
      G4Exception( "GateFictitiousVoxelMapParameterized::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException,  "Can not complete placement update.");
    }
  }
  else {
    if (theListOfOwnPhysVolume.size()) {
      G4cout  << "[GateVVolume('" << GetObjectName() << "')::ConstructOwnPhysicalVolume]:" << G4endl
      	      << "Attempting to create new placements without having emptied the vector of placements!!!" << G4endl;
      G4Exception( "GateFictitiousVoxelMapParameterized::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException, "Can not complete placement creation.");
    }
  }
 
  // We now have a queue of placements: create new volumes or update the positions of existing volumes
  // based on the content of this queue

  size_t QueueSize = pQueue->size();

  for (size_t copyNumber=0; copyNumber<QueueSize ; copyNumber++) {
     
      // Extract a combination of a rotation matrix and of a translation vector from the queue
      GatePlacement placement = pQueue->pop_front();
      G4RotationMatrix rotationMatrix = placement.first;
      G4ThreeVector position = placement.second;
     

      // If the rotation is not null, derive a dynamic rotation matrix
      G4RotationMatrix *newRotationMatrix = (rotationMatrix.isIdentity()) ? 0 : new G4RotationMatrix(rotationMatrix);
          
      pOwnPhys = GetPhysicalVolume(copyNumber);

      // Check if the physical volume exist when the geometry
      // is updating
      if (flagUpdateOnly && !pOwnPhys){ 
        G4cout << " Physical volume " << GetPhysicalVolumeName() << " does not exist!" << G4endl; 
        G4Exception( "GateFictitiousVoxelMapParameterized::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException, "Failed to construct the volume!");
      }

      if (flagUpdateOnly)
      {
        // Update physical volume
        //----------------------------------------------------------------  
        pOwnPhys = GetPhysicalVolume(copyNumber);
   
        // Set the translation vector for this physical volume
        pOwnPhys->SetTranslation(position);
     
        // Set the rotation matrix for this physical volume
        if (pOwnPhys->GetRotation())
          delete pOwnPhys->GetRotation();
      
        pOwnPhys->SetRotation(newRotationMatrix);
    
        GateMessage("Geometry", 3,"@  " << GetPhysicalVolumeName() << " has been updated." << G4endl;);
    
      }
      else
      {
        // Place new physical volume
        // Mofifs Seb JAN 23/03/2009 
        G4VPhysicalVolume* thePhysicalVolume = m_voxelInserter->GetParameterization()->GetPhysicalContainer();
        //PushPhysicalVolume(pOwnPhys);
        PushPhysicalVolume(thePhysicalVolume);
      }
  }//end for
}

std::string GateFictitiousVoxelMapParameterized::Double2String ( G4double d ) const
{
	std::stringstream ss;
	ss << d;
	return ss.str();
}
