/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifndef GateVVolume_h
#define GateVVolume_h 1

#include "GateClockDependent.hh"
#include "GateObjectChildList.hh"
#include "GateVolumePlacement.hh"
#include "GateMessageManager.hh"

#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "G4VisAttributes.hh"
#include "G4Box.hh"

#include "globals.hh"
#include <vector>
#include <map>

class GateVActor;
class G4Material;
class G4VSensitiveDetector;
class GateVolumeManager;
class GateObjectChildList;
class GateObjectRepeaterList;
class GateMaterialDatabase;
class GateVolumeMessenger;
class GateVolumePlacementMessenger;
class GateActorManager;
class GateMultiSensitiveDetector;
#ifdef GATE_USE_OPTICAL
class GateSurfaceList;
#endif

//-------------------------------------------------------------------------------------------------
class GateVVolume : public GateClockDependent
{
public :

  GateVVolume(const G4String& itsName,
	      G4bool acceptsChildren=true,
	      G4int depth=0);

  virtual ~GateVVolume();
  virtual G4VPhysicalVolume* Construct(G4bool flagUpdateOnly = false);
  virtual void ConstructGeometry(G4LogicalVolume*, G4bool);
  virtual void DestroyGeometry();
  virtual void  DestroyOwnPhysicalVolumes();

  //! Pure virtual method (to be implemented in sub-classes)
  //! Must return an value for the half-size of a volume along an axis (X=0, Y=1, Z=2)
  virtual G4double GetHalfDimension(size_t axis)=0;

  //! Print to stdout a description of the inserter
  virtual void Describe(size_t indent=0);

  //! You should override this method in sub-class if the described
  //! volume contains sub-volumes in order that actor associated with
  //! the volume are propagated to the sub-volumes. This is your
  //! responsability to do that (see GateImageNestedParametrisedVolume
  //! for exemple).
  virtual void PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector *) {}

  // Use to propagate global SD (such as PhantomSD) to child logical
  // volume. Do nothing by default. Complex volume (such as voxelized)
  // must implement this function to set the sensitive detector to
  // their sub-logical volume.
  virtual void PropagateGlobalSensitiveDetector() {}

protected :
  //! Pure virtual method, will be defined in concrete
  //! classes GateBox, GateCylinder ...
  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool)=0;
  virtual void ConstructOwnPhysicalVolume(G4bool flagUpdateOnly);

  inline virtual void PushPhysicalVolume(G4VPhysicalVolume* volume)
  { theListOfOwnPhysVolume.push_back(volume);}

  virtual void DestroyOwnSolidAndLogicalVolume()=0;

public :
  //! Attach the creator to a new inserter
  inline virtual void SetCreator(GateVVolume* anCreator)
  { m_creator = anCreator;}

  //! Return the inserter to which the creator is attached
  inline virtual GateVVolume* GetCreator() const
  { return m_creator;}

  inline virtual void SetParentVolume(GateVVolume * p) { mParent = p; }
  GateVVolume * GetParentVolume() { return mParent; }

  //! Returns the inserter's mother-list
  virtual inline GateObjectChildList* GetMotherList() const     	{ return pMotherList;}

  //! Returns the child-list
  GateObjectChildList *GetTheChildList()                             {return pChildList;}

  //! Return the surface-list object
#ifdef GATE_USE_OPTICAL
  inline virtual GateSurfaceList* GetSurfaceList() const
  { return m_surfaceList;}
#endif

  //! Sets the inserter's mother
  virtual inline void SetMotherList(GateObjectChildList* motherList) { pMotherList = motherList;}

  //! Returns the inserter's mother-creator
  virtual inline GateVVolume* GetMotherCreator() const
  { return pMotherList ? pMotherList->GetCreator() : 0 ;}

  //! Returns the list of movements
  virtual inline GateObjectRepeaterList* GetMoveList() const { return m_moveList;}

  //! Returns the list of repeaters
  virtual inline GateObjectRepeaterList* GetRepeaterList() const { return m_repeaterList;}

  virtual GateVolumePlacement* GetVolumePlacement() const;

public :

  //! Return the name used or to be used for the solid
  inline virtual const G4String& GetSolidName() const
  { return mSolidName; }

  //! Set the name of the material to be used for the volume
  inline virtual void SetMaterialName(const G4String& val)
  { mMaterialName = val; AutoSetColor(); DefineOwnMaterials(); }

  //! Return the name of the material used or to be used for the volume
  inline virtual const G4String& GetMaterialName() const
  { return mMaterialName;}

  //! Return the name of the material used or to be used for the volume
  inline virtual const G4Material* GetMaterial() const
  { return pOwnMaterial;}

  //! Return the name used or to be used for the logical volume
  inline virtual const G4String& GetLogicalVolumeName() const
  { return mLogicalVolumeName;}

  //! Returns one of the physical volumes created by its copy number
  virtual inline G4VPhysicalVolume* GetPhysicalVolume(size_t copyNumber) const
  { return (copyNumber<theListOfOwnPhysVolume.size()) ? theListOfOwnPhysVolume[copyNumber] : 0; }

  //! Return a pointer to the physical volume
  inline virtual G4VPhysicalVolume* GetPhysicalVolume() const
  { return pOwnPhys;}

  //! Return a pointer to the logical volume
  inline virtual G4LogicalVolume* GetLogicalVolume() const
  { return pOwnLog;}

  //! Return the name used or to be used for the logical volume
  inline virtual const G4String& GetPhysicalVolumeName() const
  { return mPhysicalVolumeName;}

  //! Returns the number of physical volumes created by the inserter
  virtual inline G4int GetVolumeNumber() const  	      	      	{ return theListOfOwnPhysVolume.size(); }

  //! Returns the mother logical volume for the inserter's physical volumes
  virtual inline G4LogicalVolume* GetMotherLogicalVolume() const	{ return pMotherLogicalVolume;}

  //! Return the name used or to be used for the mother logical volume
  inline virtual const G4String& GetLogicalMotherVolumeName() const
  { return mLogicalMotherVolumeName;}

  //! Return a pointer to the visibility attributes used or to be used for the volume
  inline virtual G4VisAttributes* GetVisAttributes() const { return pOwnVisAtt;}

  //! Compute the name to be used for solids based on the creator name
  static inline G4String MakeSolidName(const G4String& name)
  { return name + mTheSolidNameTag;}
  //! Compute the name to be used for logical volumes based on the creator name
  static inline G4String MakeLogicalVolumeName(const G4String& name)
  { return name + mTheLogicalVolumeNameTag;}
  //! Compute the name to be used for physical volumes based on the creator name
  static inline G4String MakePhysicalVolumeName(const G4String& name)
  { return name + mThePhysicalVolumeNameTag;}


  //! Define the sensitive detector to which the logical volume should be attached
  virtual inline void AttachSD(G4VSensitiveDetector* val)
  { m_sensitiveDetector = val; }

  //! Tell the creator that the logical volume should be attached to the crystal-SD
  virtual void AttachCrystalSD() ;

  virtual void AttachCrystalSDnoSystem();

  //! Tell the creator that the logical volume should be attached to the phantom-SD
  virtual void AttachPhantomSD() ;

  void AttachARFSD(); /* PY Descourt 08/09/2009 */

  void DumpVoxelizedVolume(G4ThreeVector);
  void SetDumpPath(G4String b) { mDumpPath = b; }

  virtual void AttachOutputToVolume();

  virtual G4bool CheckOutputExistence();

  // Origin (coordinate of the corner)
  void SetOrigin(const G4ThreeVector & i);
  inline G4ThreeVector GetOrigin() const { return m_origin; }

protected :

  //! This method retrieves a material from its name, and stores a pointer to this
  //! material into m_own_material, the method is called in GateVolumeMessenger
  virtual void DefineOwnMaterials();


  //! Method automatically called to color-code the object when its material changes.
  virtual void AutoSetColor();

protected :

  //! Name to be given to the solid
  G4String mSolidName;

  //! Name of the material to be used to construct the volume
  G4String mMaterialName;

  //! Material used for the volume
  G4Material* pOwnMaterial;

  //! Name given to the logical volume
  G4String mLogicalVolumeName;

  //! Name given to the mother logical volume
  G4String mLogicalMotherVolumeName;

  //! Physical volume name
  G4String mPhysicalVolumeName;


  //! Physical volume
  G4VPhysicalVolume* pOwnPhys;

  //! Logical volume
  G4LogicalVolume* pOwnLog;


  //! Object visualisation attribute object.
  //! It is passed to the logical volume each time the logical volume is created
  G4VisAttributes* pOwnVisAtt;


  GateObjectChildList* pMotherList;

  std::vector<G4VPhysicalVolume*> theListOfOwnPhysVolume;

  //! childen-list object
  GateObjectChildList* pChildList;

  //! surface list
#ifdef GATE_USE_OPTICAL
  GateSurfaceList* m_surfaceList;
#endif

  //!< List of repeaters
  GateObjectRepeaterList*  m_repeaterList;

  //!< List of movements
  GateObjectRepeaterList*   	  m_moveList;

  //! Mother logical volume
  G4LogicalVolume* pMotherLogicalVolume;

  //! Creators handled by the creator
  GateVVolume* m_creator;


  //! Pointer to a sensitive detector to be passed to the volume when it is constructed
  G4VSensitiveDetector* m_sensitiveDetector;

  GateVActor * pActor;

  // Some volume can have an origin, store it.
  G4ThreeVector m_origin;

private :

  //! Tag added to names to create solid names
  static const G4String mTheSolidNameTag;

  //! Tag added to names to create logical volume names
  static const G4String mTheLogicalVolumeNameTag;

  //! Tag added to names to create physical volume names
  static const G4String mThePhysicalVolumeNameTag;

  //! Translation vector
  G4ThreeVector m_translation;

  //! Rotation matrix
  G4RotationMatrix newRotationMatrix;

  //! Rotation axis (dimensionless vector)
  G4ThreeVector m_rotationAxis;

  GateVolumeMessenger* pMessenger;

  GateVVolume * mParent;

  G4String mDumpPath;
};
//-------------------------------------------------------------------------------------------------

#define FCT_FOR_AUTO_CREATOR_VOLUME(CLASS)				\
  static GateVVolume *make_volume(const G4String& itsName,		\
				  G4bool itsFlagAcceptChildren, G4int depth){return new CLASS(itsName, \
											      itsFlagAcceptChildren, depth); };

#define MAKE_AUTO_CREATOR_VOLUME(NAME,CLASS)				\
  class NAME##Creator {							\
  public:								\
  NAME##Creator() {							\
    GateVolumeManager::GetInstance()->theListOfVolumePrototypes[#NAME]= CLASS::make_volume; } }; \
  static NAME##Creator VolumeCreator##NAME;

//-------------------------------------------------------------------------------------------------
#endif
