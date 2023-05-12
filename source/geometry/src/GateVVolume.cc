/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateVVolume.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4RegionStore.hh"
#include "G4VisAttributes.hh"
#include "G4Region.hh"
#include "G4VoxelLimits.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"

#include "GateVolumeMessenger.hh"
#include "GateBox.hh"
#include "GateObjectChildList.hh"
#include "GateObjectRepeaterList.hh"
#include "GateVolumePlacement.hh"
#include "GateMaterialDatabase.hh"
#include "GateDetectorConstruction.hh"
#include "GateObjectStore.hh"
#include "GateVolumePlacement.hh"
#include "GatePlacementQueue.hh"
#include "GateTools.hh"
#include "GateActorManager.hh"
#include "GateVActor.hh"
#include "GateOutputMgr.hh"
#include "GateMessageManager.hh"
#include "GateImage.hh"

#include "GateSystemListManager.hh"

#ifdef GATE_USE_OPTICAL
#include "GateSurfaceList.hh"
#endif

#include "globals.hh"
#include <vector>
#include <fstream>
#include "GateARFSD.hh"
#include "GateDetectorConstruction.hh"

class G4Material;

// Tag added to names to create solid names
const G4String GateVVolume::mTheSolidNameTag = "_solid";
// Tag added to names to create logical volume names
const G4String GateVVolume::mTheLogicalVolumeNameTag = "_log";
//! Tag added to names to create physical volume names
const G4String GateVVolume::mThePhysicalVolumeNameTag = "_phys";

//---------------------------------------------------------------------------------------
//Constucteur
GateVVolume::GateVVolume(const G4String &itsName,
                         G4bool acceptsChildren,
                         G4int /*depth*/)
    : GateClockDependent(itsName, acceptsChildren),
      mSolidName(MakeSolidName(itsName)),
      pOwnMaterial(0),
      mLogicalVolumeName(MakeLogicalVolumeName(itsName)),
      mPhysicalVolumeName(MakePhysicalVolumeName(itsName)),
      pOwnPhys(0),
      pOwnLog(0),
      pOwnVisAtt(0),
      pMotherList(0),
      theListOfOwnPhysVolume(0),
      pChildList(0),
      m_repeaterList(0),
      m_moveList(0),
      pMotherLogicalVolume(0),
      m_creator(0),
      m_sensitiveDetector(0),
      mParent(0),
      mDumpPath("") {
    SetCreator(this);

    // Create a new vis-attributes object
    pOwnVisAtt = new G4VisAttributes();

    // Set the color based on the current material name
    AutoSetColor();

    // Create a new child-list object
    pChildList = new GateObjectChildList(this, acceptsChildren);

    // Create a new surface list object
#ifdef GATE_USE_OPTICAL
    m_surfaceList = new GateSurfaceList(this, acceptsChildren);
#endif

    // Attach a repeater list
    m_repeaterList = new GateObjectRepeaterList(this, GetObjectName() + "/repeaters", "repeater");

    // Attach a move list
    m_moveList = new GateObjectRepeaterList(this, GetObjectName() + "/moves", "move");

    // Insert a GateVolumePlacement into the move-list: that's the default move used to defined
    // the position and orientation of a volume
    m_moveList->AppendObjectRepeater(new GateVolumePlacement(this, GetObjectName() + "/placement"));

    // Attach volume creator messenger
    pMessenger = new GateVolumeMessenger(this);

    // Init origin to MAX DOUBLE
    m_origin = G4ThreeVector(DBL_MAX, DBL_MAX, DBL_MAX);

    // Register with the creator-store
    GateObjectStore::GetInstance()->RegisterCreator(this);
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Destructeur
GateVVolume::~GateVVolume() {

    delete pMessenger;
    delete pOwnVisAtt;

    //delete pOwnMaterial;
    //delete pMaterial;

    if (m_repeaterList)
        delete m_repeaterList;
    if (m_moveList)
        delete m_moveList;

#ifdef GATE_USE_OPTICAL
    // Delete the surface-list object
    delete m_surfaceList;
#endif

    delete pChildList;

    // Unregister from the creator-store
    GateObjectStore::GetInstance()->UnregisterCreator(this);

    // CAUTION: Note that we don't delete the object geometry
    // (logical volume, vector of physical volumes, visualisation attributes...)
    // Normally, an object creator should never be destroyed
    // while the volumes it handles are alive,
    // but if this were the case, these volumes are left to survive.
    // In that case, these volumes become orphans with regards to the
    // creator tree.
}
//----------------------------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVVolume::SetOrigin(const G4ThreeVector &i) {
    m_origin = i;
    GateMessage("Volume", 5, "Origin = " << m_origin << Gateendl);
}

//--------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Construct the world volume
G4VPhysicalVolume *GateVVolume::Construct(G4bool flagUpdateOnly) {
    GateMessage("Geometry", 4, "GateVVolume::Construct " << GetObjectName() << Gateendl);

    // Box volume construction
    ConstructGeometry(0, flagUpdateOnly);
    return theListOfOwnPhysVolume[0];
}
//---------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------
// Construction de world volume and all children
void GateVVolume::ConstructGeometry(G4LogicalVolume *mother_log, G4bool flagUpdateOnly) {
    GateMessage("Geometry", 7,
                "GateVVolume::ConstructGeometry -- begin ; flagUpdateOnly = " << flagUpdateOnly << Gateendl;);
    pMotherLogicalVolume = mother_log;

    if (!pOwnMaterial) {
        //GateMessage("Geometry",0," The material of the volume " << GetObjectName() << " is not defined.");
        GateError("The material of the volume " << GetObjectName() << " is not defined.");
        //return;
    }

    //  volume construction
    pOwnLog = ConstructOwnSolidAndLogicalVolume(pOwnMaterial, flagUpdateOnly);

    // Propagate Sensitive Detector if needed (see explanation in .hh)
    PropagateGlobalSensitiveDetector();

    GateMessage("Geometry", 8,
                " " << GetSolidName() << " and " << GetLogicalVolumeName() << " volumes have been constructed.\n");
    GateDebugMessageInc("Cuts", 9, "-- Constructing region for volume = " << GetObjectName() << Gateendl);

    if (GetLogicalVolumeName() != "world_log" && GetLogicalVolumeName().find("Voxel_log") == G4String::npos) {
        // Construct and add the region
        GateMessage("Cuts", 9, "- Building associated region " << GetObjectName() << Gateendl);
        G4Region *aRegion = G4RegionStore::GetInstance()->FindOrCreateRegion(GetObjectName());
        //G4Region* aRegion = new G4Region(GetObjectName());
        pOwnLog->SetRegion(aRegion);
        aRegion->AddRootLogicalVolume(pOwnLog);
    }
    //  PropagateRegionToChild();
    GateDebugMessageDec("Cuts", 9, "Region constructed\n");
    //----------------------------------------------------------------------------------------

    // Attach a sensitive detector as required
    if (m_sensitiveDetector) {
        pOwnLog->SetSensitiveDetector(m_sensitiveDetector);
        GateMessage("Geometry", 7, "A sensitive detector has been attached to " << GetObjectName() << " volume.\n");
    }

    // Set my visualisation attributes
    pOwnLog->SetVisAttributes(pOwnVisAtt);

    GateMessage("Geometry", 7, " 2 : theListOfOwnPhysVolume.size =  " << theListOfOwnPhysVolume.size() << Gateendl;);
    GateMessage("Geometry", 7, " --> Object " << GetObjectName() << " has been created.\n");

    // Construct all children
    pChildList->ConstructChildGeometry(pOwnLog, flagUpdateOnly);

    ConstructOwnPhysicalVolume(flagUpdateOnly);

    GateMessage("Geometry", 7,
                " GateVVolume::ConstructGeometry -- end ; flagUpdateOnly = " << flagUpdateOnly << Gateendl;);


    // If the origin has not been set (when read image file), we set it
    // at the center of the object. This origin is only used 1) when
    // TranslateAtThisIsocenter is set or 2) for some image actor that
    // copy this origin to the output file.
    if (m_origin[0] == DBL_MAX) {
        G4VoxelLimits limits;
        G4double min, max;
        G4AffineTransform at;
        double size[3];
        GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);
        size[0] = max - min;
        GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, min, max);
        size[1] = max - min;
        GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);
        size[2] = max - min;
        m_origin = G4ThreeVector(-size[0] / 2.0, -size[1] / 2.0, -size[2] / 2.0);
    }
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Construct physical volume
void GateVVolume::ConstructOwnPhysicalVolume(G4bool flagUpdateOnly) {
    // Store the volume default position into a placement queue
    GatePlacementQueue motherQueue;
    motherQueue.push_back(GatePlacement(G4RotationMatrix(), G4ThreeVector()));

    GatePlacementQueue *pQueue = &motherQueue;

    // Have the start-up position processed by the move list
    if (m_moveList) {
        GateMessage("Move", 5, "Compute placements of moveList for " << GetSolidName() << "\n");
        pQueue = m_moveList->ComputePlacements(pQueue);
    }
    // Have the volume's current position processed by the repeater list
    if (m_repeaterList) {
        GateMessage("Repeater", 5, "Compute placements of repeaterList for " << GetSolidName() << "\n");
        pQueue = m_repeaterList->ComputePlacements(pQueue);
    }

    GateMessage("Geometry", 6,
                GetObjectName() << " theListOfOwnPhysVolume.size  = " << theListOfOwnPhysVolume.size() << Gateendl;);
    GateMessage("Geometry", 6, GetObjectName() << " pQueue->size() = " << pQueue->size() << Gateendl;);

    // Do consistency checks
    if (flagUpdateOnly) {
        if (pQueue->size() != theListOfOwnPhysVolume.size()) {
            G4cout << "[GateVVolume('" << GetObjectName() << "')::ConstructOwnPhysicalVolume]:\n"
                   << "The size of the placement queue (" << pQueue->size() << ") is different from \n"
                   << "the number of physical volumes to update (" << theListOfOwnPhysVolume.size() << ")!!!\n";
            G4Exception("GateVVolume::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException,
                        "Can not complete placement update.");
        }
    } else {
        if (theListOfOwnPhysVolume.size()) {
            G4cout << "[GateVVolume('" << GetObjectName() << "')::ConstructOwnPhysicalVolume]:\n"
                   << "Attempting to create new placements without having emptied the vector of placements!!!\n";
            G4Exception("GateVVolume::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException,
                        "Can not complete placement creation.");
        }
    }

    // We now have a queue of placements: create new volumes or update the positions of existing volumes
    // based on the content of this queue

    size_t QueueSize = pQueue->size();

    //This for cycle is why we need an iterable queue.
    for (size_t copyNumber = 0; copyNumber < QueueSize; copyNumber++) {

        // Extract a combination of a rotation matrix and of a translation vector from the queue
        GatePlacement placement = pQueue->pop_front();
        G4RotationMatrix rotationMatrix = placement.first;
        G4ThreeVector position = placement.second;


        // If the rotation is not null, derive a dynamic rotation matrix
        G4RotationMatrix *newRotationMatrix = (rotationMatrix.isIdentity()) ? 0 : new G4RotationMatrix(rotationMatrix);

        //     GateMessage("Geometry", 5, " copyNumber = " << copyNumber << " flagUpdateOnly = " << flagUpdateOnly << " pOwnPhys exists ? = "
        //                 << pOwnPhys << " m_repeaterList = " <<m_repeaterList<< Gateendl;);

        pOwnPhys = GetPhysicalVolume(copyNumber);

        // Check if the physical volume exist when the geometry
        // is updating
        if (flagUpdateOnly && !pOwnPhys) {
            G4cout << " Physical volume " << GetPhysicalVolumeName() << " does not exist!\n";
            G4Exception("GateVVolume::ConstructOwnPhysicalVolume", "ConstructOwnPhysicalVolume", FatalException,
                        "Failed to construct the volume!");
        }

        if (flagUpdateOnly) {
            // Update physical volume
            //----------------------------------------------------------------
            pOwnPhys = GetPhysicalVolume(copyNumber);

            // Set the translation vector for this physical volume
            pOwnPhys->SetTranslation(position);

            // Set the rotation matrix for this physical volume
            if (pOwnPhys->GetRotation())
                delete pOwnPhys->GetRotation();

            pOwnPhys->SetRotation(newRotationMatrix);

            GateMessage("Geometry", 6, GetPhysicalVolumeName() << "[" << copyNumber << "] has been updated.\n";);

        } else {

            // Place new physical volume
            //---------------------------------------------------------------

            pOwnPhys = new G4PVPlacement(newRotationMatrix,        // rotation with respect to its mother volume
                                         position,                 // translation with respect to its mother volume
                                         pOwnLog,                  // the assiated logical volume
                                         GetPhysicalVolumeName(),  // physical volume name
                                         pMotherLogicalVolume,     // the mother logical volume
                                         false,                    // for future use,, can be set to false
                                         copyNumber,               // copy number
                                         false);                   // false/true = no/yes overlap check triggered

            PushPhysicalVolume(pOwnPhys);

            GateMessage("Geometry", 6, GetPhysicalVolumeName() << "[" << copyNumber << "] has been constructed.\n";);

        }

    }//end for

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Tell the creator that the logical volume should be attached to the crystal-SD
void GateVVolume::AttachCrystalSD() {
    /*
      if (!CheckOutputExistence()){
      // Add OutputMgr output actor to theListOfActors
      GateActorManager::GetInstance()->GateActorManager::AddActor("OutputMgr", "output", 0);
      AttachOutputToVolume();
      }
      else
      AttachOutputToVolume();
    */
    // Retrieve the crystal-SD pointer from the detector-construction
   // GateCrystalSD *crystalSD = GateDetectorConstruction::GetGateDetectorConstruction()->GetCrystalSD();


    // OK GND 2022
    //----
	G4SDManager* SDman = G4SDManager::GetSDMpointer();
    G4String crystalSDname = GetObjectName();// GateCrystalSD::GetCrystalCollectionName();//check GateCrystalSD class for hardcoded value
        //
    GateCrystalSD *crystalSD = new GateCrystalSD(crystalSDname);
    SDman->AddNewDetector(crystalSD);
    //----

    // Check whether this attachement is allowed or forbidden
    if (crystalSD->PrepareCreatorAttachment(this)) {
        G4cout << "[GateVVolume::AttachCrystalSD]:\n"
               << "Can not attach crystalSD!\n";
        return;
    }

    // If the attachement is allowed, store the crystal-SD pointer
    m_sensitiveDetector = crystalSD;

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Tell the creator that the logical volume should be attached to the crystal-SD
void GateVVolume::AttachCrystalSDnoSystem() {


    // OK GND 2022
    //----
	G4SDManager* SDman = G4SDManager::GetSDMpointer();
	G4String crystalSDname = GetObjectName();// GateCrystalSD::GetCrystalCollectionName();//check GateCrystalSD class for hardcoded value
        //
    GateCrystalSD *crystalSD = new GateCrystalSD(crystalSDname);
    SDman->AddNewDetector(crystalSD);
    //----
    // If the attachement is allowed, store the crystal-SD pointer
    m_sensitiveDetector = crystalSD;


    GateSystemListManager* sysList=GateSystemListManager::GetInstance();
    sysList->SetIsAnySystemDefined(false);
  //  pOwnLog->SetSensitiveDetector(m_sensitiveDetector);

  
    GateWarning("You use /attachCrystalSDnoSystem option! This means that some of Digitizer Modules will not work!\n" 
                "Here is the list:\n" 
                "+ Readout \n"
		"+ DeadTime \n"
                "+ PileUp \n");


}
//----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
// Tell the creator that the logical volume should be attached to the phantom-SD
void GateVVolume::AttachPhantomSD() {
    /*  if (!CheckOutputExistence()){
    // Add OutputMgr output actor to theListOfActors
    GateActorManager::GetInstance()->GateActorManager::AddActor("OutputMgr", "output", 0);
    AttachOutputToVolume();

    }
    else
    AttachOutputToVolume();
    */
    // Retrieve the phantom-SD pointer from the detector-construction, and store this pointer //
    GatePhantomSD *phantomSD = GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD();

    // If the attachement is allowed, store the crystal-SD pointer
    m_sensitiveDetector = phantomSD;
}
//------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4bool GateVVolume::CheckOutputExistence() {
    std::vector<GateVActor *> theList;
    theList = GateActorManager::GetInstance()->GateActorManager::ReturnListOfActors();

    std::vector<GateVActor *>::iterator sit;
    for (sit = theList.begin(); sit != theList.end(); ++sit) {
        if ((*sit)->GetObjectName() == "output") {
            return true;
        }
    }

    return false;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
void GateVVolume::AttachOutputToVolume() {

    // Attach volume to the actor "output"
    std::vector<GateVActor *> theList;
    theList = GateActorManager::GetInstance()->GateActorManager::ReturnListOfActors();

    std::vector<GateVActor *>::iterator sit;

    for (sit = theList.begin(); sit != theList.end(); ++sit) {
        if ((*sit)->GetObjectName() == "output") {
            (*sit)->GateVActor::SetVolumeName(GetObjectName());
            (*sit)->GateVActor::AttachToVolume(GetObjectName());
        }
    }

}
//------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Destroy all geometry
void GateVVolume::DestroyGeometry() {
    GateMessage("Geometry", 5, "GateVVolume::DestroyGeometry Volume " << GetObjectName()
                                                                      << " is going to be destroyed.\n");
    while (!theListOfOwnPhysVolume.empty()) {
        // Get the last physical volume
        G4VPhysicalVolume *lastVolume = theListOfOwnPhysVolume.back();

        G4int n = 0;
        n = lastVolume->GetCopyNo();

        if (GetMotherLogicalVolume())
            GetMotherLogicalVolume()->RemoveDaughter(lastVolume);

        // Destroy the volume rotation if required
        if (lastVolume->GetRotation())
            delete lastVolume->GetRotation();

        // Destroy the physical volume
        delete lastVolume;

        // Remove the volume from the physical-volume vector
        theListOfOwnPhysVolume.erase(theListOfOwnPhysVolume.end() - 1);
        pOwnPhys = 0;

        pChildList->DestroyChildGeometry();

        // Destroy the solid and logical volumes
        DestroyOwnSolidAndLogicalVolume();

        pOwnLog = 0;

        GateMessage("Geometry", 5,
                    "GateVVolume :: Destroy geometry of object " << GetObjectName() << " with copy number " << n
                                                                 << ".\n";);
    }
}
//-----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Destroy all geometry
void GateVVolume::DestroyOwnPhysicalVolumes() {
    //  GateMessage("Geometry", 0, " Volumes are going to be destroyed.\n");
    while (!theListOfOwnPhysVolume.empty()) {
        // Get the last physical volume
        G4VPhysicalVolume *lastVolume = theListOfOwnPhysVolume.back();

        if (GetMotherLogicalVolume())
            GetMotherLogicalVolume()->RemoveDaughter(lastVolume);

        // Destroy the volume rotation if required
        if (lastVolume->GetRotation())
            delete lastVolume->GetRotation();

        // Destroy the physical volume
        delete lastVolume;

        // Remove the volume from the physical-volume vector
        theListOfOwnPhysVolume.erase(theListOfOwnPhysVolume.end() - 1);
    }
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Method automatically called before the construction of a volume.
// It retrieves a material from its name, and stores a pointer to this material into pOwnMaterial
void GateVVolume::DefineOwnMaterials() {

    // Retrieve the material pointer from the material database
    pOwnMaterial = theMaterialDatabase.GetMaterial(mMaterialName);
    // If we could not get the material, it is unsafe to proceed: abort!
    if (!pOwnMaterial)
        G4Exception("GateVVolume::DefineOwnMaterials", "DefineOwnMaterials", FatalException,
                    "GateVVolume::DefineOwnMaterials: \n"
                    "Could not find a material needed for the construction of the scene!\n"
                    "Computation aborted!!!");

}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Returns the inserter's placement
GateVolumePlacement *GateVVolume::GetVolumePlacement() const {
    return (GateVolumePlacement *) ((m_moveList) ? m_moveList->GetRepeater(0) : 0);
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Method automatically called to color-code the object when its material changes.
void GateVVolume::AutoSetColor() {
    // We compare the material name with a hard-coded (sorry!) list
    // If the name is recognised, we set its typical color
    // If not, we pick gray by default

    pOwnVisAtt->SetForceWireframe(false);
    if (mMaterialName == "worldDefaultAir") {
        pOwnVisAtt->SetColor(G4Colour(1.0, 1.0, 1.0));
        pOwnVisAtt->SetForceWireframe(true);
    }
    else if (mMaterialName == "G4_AIR")
        pOwnVisAtt->SetColor(G4Colour(1.0, 1.0, 1.0));
    else if (mMaterialName == "Air")
        pOwnVisAtt->SetColor(G4Colour(1.0, 1.0, 1.0));
    else if (mMaterialName == "G4_WATER")
        pOwnVisAtt->SetColor(G4Colour(0.0, 1.0, 1.0));
    else if (mMaterialName == "Water")
        pOwnVisAtt->SetColor(G4Colour(0.0, 1.0, 1.0));
    else if ((mMaterialName == "NaI") ||
             (mMaterialName == "PWO") ||
             (mMaterialName == "BGO") ||
             (mMaterialName == "LSO") ||
             (mMaterialName == "GSO") ||
             (mMaterialName == "LuAP") ||
             (mMaterialName == "LuYAP-70") ||
             (mMaterialName == "LuYAP-80") ||
             (mMaterialName == "Silicon") ||
             (mMaterialName == "Germanium") ||
             (mMaterialName == "YAP") ||
             (mMaterialName == "Scinti-C9H10"))
        pOwnVisAtt->SetColor(G4Colour(1.0, 1.0, 0.0));
    else if ((mMaterialName == "Lead") ||
             (mMaterialName == "Tungsten") ||
             (mMaterialName == "Bismuth"))
        pOwnVisAtt->SetColor(G4Colour(0.0, 0.0, 1.0));
    else if (mMaterialName == "Aluminium")
        pOwnVisAtt->SetColor(G4Colour(0.0, 1.0, 0.0));
    else if (mMaterialName == "Breast")
        pOwnVisAtt->SetColor(G4Colour(1.0, 0.0, 1.0));
    else
        pOwnVisAtt->SetColor(G4Colour(0.5, 0.5, 0.5));
}
//-----------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// Print to stdout a description of the inserter
void GateVVolume::Describe(size_t indent) {
    G4cout << GateTools::Indent(indent) << "-----------------------------------------------\n";
    G4cout << Gateendl;
    if (m_moveList)
        m_moveList->DescribeRepeaters(indent);
    if (m_repeaterList)
        m_repeaterList->DescribeRepeaters(indent);
    G4cout << GateTools::Indent(indent) << "-----------------------------------------------"
           << Gateendl << Gateendl;
}
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
/* PY Descourt 08/09/2009 */
void GateVVolume::AttachARFSD() {
    // Retrieve the crystal-SD pointer from the detector-construction
    GateARFSD *arfSD = GateDetectorConstruction::GetGateDetectorConstruction()->GetARFSD();

    // Check whether this attachement is allowed or forbidden
    if (arfSD->PrepareCreatorAttachment(this)) {
        G4cout << "[GateVObjectCreator::AttachARFSD]:\n"
               << "Can not attach ARFSD!\n";
        return;
    }

    G4cout << " GateVObjectCreator::AttachARFSD() :::: created an attachment to ARF Sensitive Detector " << arfSD
           << Gateendl;

    // If the attachement is allowed, store the crystal-SD pointer
    m_sensitiveDetector = arfSD;
}
/* PY Descourt 08/09/2009 */
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
void GateVVolume::DumpVoxelizedVolume(G4ThreeVector spacing) {
    time_t tStart, tEnd;
    time(&tStart);

    G4Navigator *navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();

    G4VPhysicalVolume *WorldVolume = navigator->GetWorldVolume();
    G4cout << "Current world volume: " << navigator->GetWorldVolume()->GetName() << G4endl;

    navigator->SetWorldVolume(GetPhysicalVolume());

    G4cout << "Current world volume: " << navigator->GetWorldVolume()->GetName() << G4endl;

    G4VoxelLimits limits;
    G4double min, max;
    G4AffineTransform at;
    G4ThreeVector size;
    GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, at, min, max);
    size.setX(max - min);
    GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, at, min, max);
    size.setY(max - min);
    GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, at, min, max);
    size.setZ(max - min);

    const G4ThreeVector resolution(std::round(size.x() / spacing.x()), std::round(size.y() / spacing.y()),
                                   std::round(size.z() / spacing.z()));

    spacing.set(size.x() / resolution.x(), size.y() / resolution.y(), size.z() / resolution.z());

    GateImageInt *image = new GateImageInt();
    image->SetResolutionAndVoxelSize(resolution, spacing);
    image->Allocate();

    GateMessage("Geometry", 0, "[GateVVolume::" << __FUNCTION__ << "] DEBUG: Voxelized image informations:" << Gateendl
                                                << "      resolution: " << resolution.x() << "," << resolution.y()
                                                << "," << resolution.z() << Gateendl
                                                << "      size      : " << size.x() << "," << size.y() << ","
                                                << size.z() << " mm" << Gateendl
                                                << "      spacing   : " << spacing.x() << "," << spacing.y() << ","
                                                << spacing.z() << " mm" << Gateendl
                                                << "      voxels    : " << image->GetNumberOfValues() << Gateendl);

    std::vector<G4String> name;

    if (mDumpPath == "")
        mDumpPath = "data/";

    G4String HU2MatPath = mDumpPath + "/" + GetPhysicalVolume()->GetName() + "-HU2Mat.txt";
    std::ofstream file(HU2MatPath, std::ofstream::out | std::ofstream::trunc);

    for (signed long int index = 0; index < image->GetNumberOfValues(); index++) {
        const G4ThreeVector doselCoordinates = image->GetVoxelCenterFromIndex(index);
        const G4VPhysicalVolume *PV = navigator->LocateGlobalPointAndSetup(doselCoordinates);

        bool OK = false;
        int nb = 0;
        for (size_t i = 0; i < name.size(); i++)
            if (name[i] == PV->GetLogicalVolume()->GetMaterial()->GetName()) {
                OK = true;
                nb = i;
            }

        if (!OK) {
            name.push_back(PV->GetLogicalVolume()->GetMaterial()->GetName());
            nb = name.size() - 1;
            std::ostringstream a, b;
            a << nb;
            b << nb + 1;
            file << a.str() << " " << a.str() << " " << PV->GetLogicalVolume()->GetMaterial()->GetName() << std::endl;
        }

        //G4cout << "Index: " << index << " , Mat name is: " << PV->GetLogicalVolume()->GetMaterial()->GetName() << " (" << nb << ")" << G4endl;

        image->SetValue(index, nb);
    }

    file.close();

    GateMessage("Geometry", 0, "[GateVVolume::" << __FUNCTION__ << "] " << HU2MatPath << " written" << Gateendl);

    std::ostringstream sx, sy, sz;
    sx << spacing.x();
    sy << spacing.y();
    sz << spacing.z();

    G4String imagePath =
        mDumpPath + "/" + GetPhysicalVolume()->GetName() + "-" + sx.str() + "-" + sy.str() + "-" + sz.str() + "mm.mhd";

    image->Write(imagePath);

    navigator->SetWorldVolume(WorldVolume);
    G4cout << "Current world volume: " << navigator->GetWorldVolume()->GetName() << G4endl;

    GateMessage("Geometry", 0, "[GateVVolume::" << __FUNCTION__ << "] " << imagePath << " written" << Gateendl);

    time(&tEnd);

    GateMessage("Geometry", 0,
                "[GateVVolume::" << __FUNCTION__ << "] DEBUG: Computing time: " << difftime(tEnd, tStart) << " s"
                                 << Gateendl);
}
//------------------------------------------------------------------------------------------
