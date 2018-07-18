/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateRegularParameterized.hh"
#include "GateGeometryVoxelImageReader.hh"
#include "GateGeometryVoxelInterfileReader.hh"
#include "GateOutputMgr.hh"
#include "GateBox.hh"
#include "GateVVolume.hh"
#include "GateVoxelOutput.hh"
#include "GateVOutputModule.hh"
#include "G4VisAttributes.hh"
#include "GatePlacementQueue.hh"
#include "GateObjectRepeaterList.hh"

///////////////////
//  Constructor  //
///////////////////

GateRegularParameterized::GateRegularParameterized(const  G4String& name,
						   G4bool acceptsChildren,
		 			   	   G4int  depth)
: GateBox(name,"Vacuum",1,1,1,acceptsChildren,depth),
  m_name(name),
  m_messenger(new GateRegularParameterizedMessenger(this)),
  m_voxelReader(0),
  m_voxelInserter(new GateRegularParam(name+"Voxel", this)),
  voxelNumber(G4ThreeVector(1,1,1)),
  voxelSize(G4ThreeVector(1,1,1))
{

  verboseLevel=0; //! Default : set to quiet
  skipEqualMaterials = 0; //! Default: because G4 bug for speed up set at 1
  GetCreator()->GetTheChildList()->AddChild(m_voxelInserter);
}


GateRegularParameterized::GateRegularParameterized(const G4String& name) :
GateBox(name,"Vacuum",1,1,1,false,false),
                               m_name(name),
                               m_messenger(new GateRegularParameterizedMessenger(this)),
                               m_voxelReader(0),
                               m_voxelInserter(new GateRegularParam(name+"Voxel", this)),
                               voxelNumber(G4ThreeVector(1,1,1)),
                               voxelSize(G4ThreeVector(1,1,1))
{
  verboseLevel=0; //! Default : set to quiet
  skipEqualMaterials = 0; //! Default: because G4 bug for speed up set at 1
  GetCreator()->GetTheChildList()->AddChild(m_voxelInserter);
}

//////////////////
//  Destructor  //
//////////////////

GateRegularParameterized::~GateRegularParameterized()
{
  delete m_messenger;
}

////////////////////
//  InsertReader  //
////////////////////

void GateRegularParameterized::InsertReader(G4String readerType)
{
  if (verboseLevel>=1) {
    G4cout << "++++ Entering GateRegularParameterized::InsertReader ..."
           << Gateendl << std::flush;
  }

  if (m_voxelReader) {
    G4cout << "GateRegularParameterized::InsertReader: voxel reader already defined\n";
    return;
  }

  if (readerType == G4String("image")){
    m_voxelReader = new GateGeometryVoxelImageReader(this);
  } else if (readerType == G4String("interfile")) {
    m_voxelReader = new GateGeometryVoxelInterfileReader(this);
  } else
    G4cout << "GateRegularParameterized::InsertReader: unknown reader type\n";

  if (verboseLevel>=1) {
    G4cout << "---- Exiting GateRegularParameterized::InsertReader ..."
           << Gateendl << std::flush;
  }
}

////////////////////
//  RemoveReader  //
////////////////////

void GateRegularParameterized::RemoveReader()
{
  if (verboseLevel>=1) {
    G4cout << "+-+- Entering GateRegularParameterized::RemoveReader ..."
           << Gateendl << std::flush;
  }
}

///////////////////////
//  AttachPhantomSD  //
///////////////////////

void GateRegularParameterized::AttachPhantomSD()
{
  if (verboseLevel>=1) {
    G4cout << "++++ Entering GateRegularParameterized::AttachPhantomSD ..."
           << Gateendl << std::flush;
  }

  m_voxelInserter->GetCreator()->AttachPhantomSD();

  if (verboseLevel>=1) {
    G4cout << "---- Exiting GateRegularParameterized::AttachPhantomSD ..."
           << Gateendl << std::flush;
  }
}

/////////////////
//  AddOutput  //
/////////////////

void GateRegularParameterized::AddOutput(G4String name)
{
  if (verboseLevel>=1) {
    G4cout << "++++ Entering GateRegularParameterized::AddOutput ..."
           << Gateendl << std::flush;
  }

  GateOutputMgr* mgr(GateOutputMgr::GetInstance());
  mgr->AddOutputModule( (GateVOutputModule*) new GateVoxelOutput(name, GetObjectName(), mgr, mgr->GetDigiMode(),this) );

  if (verboseLevel>=1) {
    G4cout << "---- Exiting GateRegularParameterized::AddOutput ..."
           << Gateendl << std::flush;
  }
}

/////////////////////////
//  ConstructGeometry  //
/////////////////////////

void GateRegularParameterized::ConstructGeometry(G4LogicalVolume* mother_log, G4bool flagUpdateOnly)
{

  if (verboseLevel>=1) {
    G4cout << "++++ Entering GateRegularParameterized::ConstructGeometry ..."
           << Gateendl
           << "     --> with : flagUpdateOnly = " << flagUpdateOnly
           << "  |  mother_log = " << mother_log
           << Gateendl << std::flush;
  }

  if (m_voxelReader)
  {
    // Get the voxel number and size from the reader
    voxelNumber = G4ThreeVector( m_voxelReader->GetVoxelNx(),
                                 m_voxelReader->GetVoxelNy(),
                                 m_voxelReader->GetVoxelNz() );
    voxelSize   = G4ThreeVector( m_voxelReader->GetVoxelSize() );
  }
  else
  {
    G4cout << "GateRegularParameterized::ConstructGeometry - Warning ! ConstructGeometry called without a reader\n" << std::flush;
    return;
  }

  GateBox* m_boxCreator;

  // Update the dimensions of the enclosing box
  m_boxCreator = dynamic_cast<GateBox*>( GetCreator() );
  m_boxCreator->SetBoxXLength( voxelNumber.x() * voxelSize.x() );
  m_boxCreator->SetBoxYLength( voxelNumber.y() * voxelSize.y() );
  m_boxCreator->SetBoxZLength( voxelNumber.z() * voxelSize.z() );

  // Update the dimensions of the voxel box

  m_boxCreator = dynamic_cast<GateBox*>(m_voxelInserter->GetCreator());
  m_boxCreator->SetBoxXLength( voxelSize.x() );
  m_boxCreator->SetBoxYLength( voxelSize.y() );
  m_boxCreator->SetBoxZLength( voxelSize.z() );

  // Proceed with the rest
  GateVVolume::ConstructGeometry(mother_log, flagUpdateOnly);

// PY. Descourt 06/02/2009 */
/*
	if ( !flagUpdateOnly )
	{
		G4RegionStore* regionstore = G4RegionStore::GetInstance();
	    G4Region* theregular_region = regionstore->GetRegion("RegularPhantomRegion",false);

		if ( theregular_region != 0 )
		delete theregular_region;

		//add region
		G4Envelope* region=new G4Region ( "RegularPhantomRegion" ); // destruction should be handled by G4RegionStore

		if ( region==NULL )
			G4Exception ( "GateRegularParameterizedInserter::ConstructGeometry: Cannot create/allocate G4Region! Aborting." );
		if ( m_pProductionCuts!=NULL )
			region->SetProductionCuts ( m_pProductionCuts );
		G4LogicalVolume* vol=GetCreator()->GetLogicalVolume();
                if ( verboseLevel>=2 )
			G4cout << "Attach " << vol->GetName() << " to regular phantom region.\n";
		region->AddRootLogicalVolume ( vol ); //m_boxCreator->GetLogicalVolume() );

	}
	* */
// PY. Descourt 06/02/2009 */


  // Visibility attributes
  G4VisAttributes* creatorVis= const_cast<G4VisAttributes*>(GetCreator()->GetLogicalVolume()->GetVisAttributes());
  creatorVis->SetForceWireframe(true);

  if (verboseLevel>=1) {
    G4cout << "---- Exiting GateRegularParameterized::ConstructGeometry ..."
           << Gateendl << std::flush;
  }
}

///////////////////////////////////
//  ConstructOwnPhysicalVolumes  //
///////////////////////////////////
void GateRegularParameterized::ConstructOwnPhysicalVolume(G4bool flagUpdateOnly)
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
//    G4cout << " *** repeaterList exists, repeaterList->ComputePlacements(pQueue)\n";
    pQueue = m_repeaterList->ComputePlacements(pQueue);}


  // Do consistency checks
  if (flagUpdateOnly && theListOfOwnPhysVolume.size()) {
    if (pQueue->size()!=theListOfOwnPhysVolume.size()) {
      G4cout  << "[GateVVolume('" << GetObjectName() << "')::ConstructOwnPhysicalVolumes]:\n"
      	      << "The size of the placement queue (" << pQueue->size() << ") is different from \n"
	      << "the number of physical volumes to update (" << theListOfOwnPhysVolume.size() << ")!!!\n";
      G4Exception( "GateRegularParameterized::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException, "Can not complete placement update.");
    }
  }
  else {
    if (theListOfOwnPhysVolume.size()) {
      G4cout  << "[GateVVolume('" << GetObjectName() << "')::ConstructOwnPhysicalVolumes]:\n"
      	      << "Attempting to create new placements without having emptied the vector of placements!!!\n";
      G4Exception( "GateRegularParameterized::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException, "Can not complete placement creation.");
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
    G4cout << " Physical volume " << GetPhysicalVolumeName() << " does not exist!\n";
    G4Exception( "GateRegularParameterized::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException,  "Failed to construct the volume!");
  }


//  if (flagUpdateOnly && pOwnPhys)
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

    GateMessage("Geometry", 3,"@  " << GetPhysicalVolumeName() << " has been updated.\n";);

    }
    else
    {
    // Place new physical volume
    // Modifs Seb JAN 23/03/2009
    G4VPhysicalVolume* thePhysicalVolume = m_voxelInserter->GetParameterization()->GetPhysicalContainer();
    //PushPhysicalVolume(pOwnPhys);
    PushPhysicalVolume(thePhysicalVolume);
  }

  }//end for
}
//----------------------------------------------------------------------------------------
