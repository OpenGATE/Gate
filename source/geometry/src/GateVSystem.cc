/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateVSystem.hh"
//LF
//#include <strstream>
#include <sstream>
//LF
#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"
#include "G4VSolid.hh"

#include "GateTools.hh"
#include "GateSystemListManager.hh"
#include "GateVVolume.hh"
#include "GateHit.hh"
#include "GateSystemComponent.hh"
#include "GateObjectChildList.hh"
#include "GateLinearRepeater.hh"
#include "GateRootDefs.hh"
#include "GateObjectRepeaterList.hh"
#include "GateVolumeID.hh"
#include "GateDetectorConstruction.hh"
#include "GateSystemListMessenger.hh"

#include "GateConfiguration.h"

#include "GateToTree.hh"

G4int GateVSystem::m_insertionOrder=-1;
//-----------------------------------------------------------------------------
/* Constructor

   itsName:      the name chosen for this system
   isWithGantry:	tells whether there is a gantry (PET) or not (SPECT)
*/    
GateVSystem::GateVSystem(const G4String& itsName,G4bool isWithGantry)
  : GateClockDependent( itsName , false ),
    m_BaseComponent(0),
    m_mainComponentDepth( isWithGantry ? 1 : 0 ) 
{
  // Next lines were added for the multi-system approach
  G4String itsOwnName = GateSystemListManager::GetInstance()->GetInsertedSystemsNames()->back();

  // Papa's Debugging : number of system for headID in case of multi-system
  G4int sysNumber = GateSystemListManager::GetInstance()->GetInsertedSystemsNames()->size();
  m_sysNumber = sysNumber;

  m_itsOwnName = itsOwnName;
  m_insertionOrder++;
  m_itsNumber = m_insertionOrder;
  
  SetBaseComponent( new GateSystemComponent("base",0,this) );
  
  // Register the system in the system-store
  GateSystemListManager::GetInstance()->RegisterSystem(this);
  

#ifdef G4ANALYSIS_USE_ROOT
  GateRootDefs::SetDefaultOutputIDNames();
#endif
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Destructor
GateVSystem::~GateVSystem() 
{
  // Delete the tree of system-components
  delete m_BaseComponent;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Set the outputID name for a depth of the tree
void GateVSystem::SetOutputIDName(char * anOutputIDName, size_t depth)
{
#ifdef G4ANALYSIS_USE_ROOT
  GateRootDefs::SetOutputIDName(anOutputIDName,depth);

#endif
  GateToTree::SetOutputIDName(m_itsNumber, anOutputIDName, depth);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/* Method overloading the base-class virtual method Describe().
   This methods prints-out a description of the system

   indent: the print-out indentation (cosmetic parameter)
*/    
void GateVSystem::Describe(size_t indent)
{
  // Call the base-class method
  GateClockDependent::Describe(indent);

  // Print-out the min and max component
  G4cout << GateTools::Indent(indent) << "Components:\n";

  // Ask for a recursive print-out of the components
  m_BaseComponent->Describe(indent);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Compute the depth of the component tree
size_t GateVSystem::GetTreeDepth() const
{
  // Ask the tree to compute its depth
  return m_BaseComponent->GetTreeDepth();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Returns a GateSystemComponent from its name
GateSystemComponent* GateVSystem::FindComponent(const G4String& componentName,G4bool silent) const
{
  // Ask the component tree to look for the component
  GateSystemComponent* aComponent = m_BaseComponent->FindSystemComponent(GetObjectName() + "/" + componentName);
  if ( (!aComponent) && (!silent) )
    G4cerr << Gateendl << "[" << GetObjectName() << "::FindComponent]:\n"
      	   << "\tCould not find the request system component '" << componentName << "'!\n";
  return aComponent;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// template to find components of a specific type
template <class C>
C* GateVSystem::FindTypedComponent(const G4String& componentName) const
{
  // Look for the component, based on its name
  GateSystemComponent* aComponent = FindComponent(componentName);
  
  // Cast to the requested type
  return dynamic_cast<C*>(aComponent);
} 
//-----------------------------------------------------------------------------
    
//-----------------------------------------------------------------------------
// Finds an array-component from its name
GateArrayComponent* GateVSystem::FindArrayComponent(const G4String& componentName) const
{
  return FindTypedComponent<GateArrayComponent>(componentName);
} 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Finds a boxcreator-component from its name
GateBoxComponent* GateVSystem::FindBoxCreatorComponent(const G4String& componentName) const
{
  return FindTypedComponent<GateBoxComponent>(componentName);
} 
//-----------------------------------------------------------------------------
 
//-----------------------------------------------------------------------------
// Finds a cylindercreator-component from its name
GateCylinderComponent* GateVSystem::FindCylinderCreatorComponent(const G4String& componentName) const
{
  return FindTypedComponent<GateCylinderComponent>(componentName);
} 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Finds a cylindercreator-component from its name
GateWedgeComponent* GateVSystem::FindWedgeCreatorComponent(const G4String& componentName) const
{
  return FindTypedComponent<GateWedgeComponent>(componentName);
} 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Generate the output-volumeID based on the information stored in the volumeID
GateOutputVolumeID GateVSystem::ComputeOutputVolumeID(const GateVolumeID& aVolumeID)
{
  // Create a new, empty output-volume ID
  GateOutputVolumeID outputVolumeID(GetTreeDepth());

  // verbose output
  if (nVerboseLevel)
    G4cout << "[" << GetObjectName() << "::ComputeOutputVolumeID]:\n"
      	   << "\tComputing the output volume ID for the hit in volume:" << aVolumeID << Gateendl; 

  // Ask the component-tree to compute the output-volume ID
  outputVolumeID[0]=ComputeSubtreeID(m_BaseComponent,aVolumeID,outputVolumeID,0);
  
  // Set the first cell to the systemID
  //Papa's Debug
  if (m_sysNumber > 1) outputVolumeID[0]=this->GetItsNumber();
  /////

  // verbose output
  if (nVerboseLevel)
    G4cout << "[" << GetObjectName() << "::ComputeOutputVolumeID]:\n"
      	   << "\tOutput volume ID is:" << outputVolumeID << Gateendl;

  return outputVolumeID;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Compute a subsection of an output-volumeID for the subtree starting from a component
G4int GateVSystem::ComputeSubtreeID(GateSystemComponent* aComponent, 
                                    const GateVolumeID& volumeID,
                                    GateOutputVolumeID& outputVolumeID,
                                    size_t depth)
{
  // Compute the ID for the current component
  G4int result = ComputeComponentID(aComponent,volumeID);

  // Ask each subtree to compute its section of the output ID until ones returns a success value
  G4int subtreeResult = -1;
  for ( size_t i=0 ; i<aComponent->GetChildNumber() ; ++i) {
    subtreeResult = ComputeSubtreeID(aComponent->GetChildComponent(i),volumeID,outputVolumeID,depth+1);
    if (subtreeResult>=0) {
      outputVolumeID[depth+1]=subtreeResult;
      break;
    }
  }
	
  // If a subtree returned a success value while the current component failed,
  // set the result to 0
  if ( (subtreeResult>=0) && (result<0) )
    result = 0;

  // If a result was found, increase the result by the number of 'younger' components
  if (result>=0) 
    result += aComponent->ComputeOutputOffset();

  // Return our result, so that our caller can know whether we succeeded or failed
  return result;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Compute a single bin of an output ID for a component 
G4int GateVSystem::ComputeComponentID(GateSystemComponent* aComponent, const GateVolumeID& volumeID)
{
  if (aComponent->GetCreator()==0)
    return -1;

  if ( aComponent == GetMainComponent() )
    return ComputeMainComponentID(aComponent,volumeID);

  // Loop on the volume's creators until we recognise the component's creator
  for (size_t depth=0; depth<volumeID.size(); ++depth)
    if ( volumeID.GetCreator(depth) == aComponent->GetCreator())
      return volumeID.GetCopyNo(depth);

  // We did not recognise the Cretaor
  return -1;
}  
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Compute a single bin of an output ID for a coincident component 
G4int GateVSystem::ComputeMainComponentID(GateSystemComponent* aComponent, const GateVolumeID& volumeID)
{
  G4int copyNo=-1;

  // Loop on the volume's creators until we recognise our own creator
  size_t depth;
  for (depth=0; depth<volumeID.size(); ++depth)
    if ( volumeID.GetCreator(depth) == aComponent->GetCreator()) {
      copyNo = volumeID.GetCopyNo(depth);
      break;
    }

  // We did not recognise the creator
  if (depth==volumeID.size()) 
    return -1;

  // OK, we found our creator, so we need to check that the copyNo agrees with the CCC convention
  GateObjectRepeaterList* repeaterlist = aComponent->GetCreator()->GetRepeaterList();

  // If there is no repeater or if the linear comes first, it's OK
  GateVGlobalPlacement* firstRepeater = repeaterlist->GetRepeater(0);
  if (!firstRepeater)
    return copyNo;
  if (dynamic_cast<GateLinearRepeater*>(firstRepeater)!=0)
    return copyNo;
    
  // OK, we assume that the angular repeater comes first
  // We must decode the copyNo then create the componentID
  G4int angularRepeatNumber = aComponent->GetAngularRepeatNumber();
  G4int linearRepeatNumber  = aComponent->GetLinearRepeatNumber();
  G4int angularIndex        = copyNo / linearRepeatNumber;
  G4int axialIndex          = copyNo % linearRepeatNumber;

  G4int componentID         =  axialIndex * angularRepeatNumber + angularIndex; 

  return componentID;
}  
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/* Check whether an creator is connected to the system
   (directly or through one of its ancestors)
      	
   anCreator: the creator we want to check
	
   returns true if the creatr belongs (directly or inderectly) to the system
*/
G4bool GateVSystem::CheckConnectionToCreator(GateVVolume* anCreator) const
{
  //  G4cout << " DEBUT CheckConnectionToCreator \n";
  
  // Loop as long as we have a valid creator
  while (anCreator)
    {
      // Ask the component tree to look for a connection with the current creator
      G4bool result = m_BaseComponent->CheckConnectionToCreator(anCreator);

      //    G4cout << " result = " << result << Gateendl;
      // A conncetion was found: return true
      if (result)
        return true;
      
      // No connection was found: move up by one step in the creator tree
      GateVVolume *motherCreator = anCreator->GetMotherCreator();
    
      //    G4cout << " motherCreator "  << Gateendl;
      anCreator = motherCreator ?  motherCreator->GetCreator() : 0;
    }
  
  // No connection was found with the creator or with its ancestors: return false
  return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

G4bool GateVSystem::CheckIfAllLevelsAreDefined()
{

    G4int systemDepth=this->GetTreeDepth();

    for (G4int i=1;i<systemDepth-1;i++)
    {
    	GateSystemComponent* comp0= (this->MakeComponentListAtLevel(i))[0][0];
    	//G4cout<< comp0->GetObjectName()<<G4endl;

    	if (!comp0->GetCreator())
    			return false;
    }

  return true;
}
//-----------------------------------------------------------------------------

G4bool GateVSystem::CheckIfEnoughLevelsAreDefined()
{
    G4int systemDepth=this->GetTreeDepth();
    G4bool has_undefined_high_level = false;
    for (G4int i=1;i<systemDepth-1;i++)
    {
      auto compList = this->MakeComponentListAtLevel(i);
    	GateSystemComponent* comp0= compList[0][0];
    	if (!comp0->GetCreator())
      {
        if (!has_undefined_high_level)
          has_undefined_high_level = true;
      }
      else if (has_undefined_high_level)
      {
        delete compList;
        return false;
      }
      delete compList;
    }

  return true;
}

//-----------------------------------------------------------------------------
GateVSystem::compList_t* GateVSystem::MakeComponentListAtLevel(G4int level) const
{
  compList_t* currentList = new compList_t;
  currentList->push_back(m_BaseComponent);
  while ( (level>0) && !currentList->empty() ){
    level--;
    compList_t*  newList = new compList_t;
    for (size_t i=0;i<currentList->size();++i){
      for (size_t ichild=0;ichild<(*currentList)[i]->GetChildNumber();++ichild){
        GateSystemComponent* comp = (*currentList)[i]->GetChildComponent(ichild);
        newList->push_back(comp);
      }
    }
    delete currentList;
    currentList = newList;
  }
  return currentList;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
size_t GateVSystem::ComputeNofElementsAtLevel(size_t level) const
{
  compList_t* currentList = MakeComponentListAtLevel(level);
  size_t ans = 0;
  for (size_t i=0;i<currentList->size();++i){
    if ( (*currentList)[i]->IsActive() ){
      size_t nofVol = (*currentList)[i]->GetVolumeNumber();
      ans += nofVol ? nofVol : 1;
    }
  }
  delete currentList;
  return ans;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
size_t GateVSystem::ComputeNofSubCrystalsAtLevel(size_t level, std::vector<G4bool>& enableList) const
{
  size_t treeDepth = GetTreeDepth();
  if (level>=treeDepth) return 0;
  if (level == treeDepth-1 ) {
    return 1;
  } else {
    size_t n=enableList[level +1] ? ComputeNofElementsAtLevel(level+1) : 1;
    size_t ans = ComputeNofSubCrystalsAtLevel(level+1,enableList)*n;

    return ans;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
size_t GateVSystem::ComputeIdFromVolID(const GateOutputVolumeID& volID,std::vector<G4bool>& enableList) const
{
  static G4bool isFirstPass=true;
  static std::vector<G4int> nofCrystalList;
  if (isFirstPass){
    isFirstPass=false;
    for (size_t i=0;i<GetTreeDepth();i++){
      G4cout<<"nofSubCrystal @ level "<<i<< Gateendl;
      nofCrystalList.push_back(ComputeNofSubCrystalsAtLevel(i,enableList));
      G4cout<<"= "<<nofCrystalList[i]<< Gateendl;
    }
  }
  size_t ans = 0;
  for (size_t i=0;i<GetTreeDepth();i++){
    if (enableList[i] && volID[i]>=0)  ans += volID[i] * nofCrystalList[i];
  }
  return ans; 
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateVolumeID* GateVSystem::MakeVolumeID(const std::vector<G4int>& numList) const
{
  GateSystemComponent* comp = GetBaseComponent();
  if (!comp) return 0;
  G4VPhysicalVolume *vol=comp->GetPhysicalVolume(0), *last_vol=vol;
  GateVolumeID* ans = new GateVolumeID;
  ans->push_back( GateVolumeSelector(GateDetectorConstruction::GetGateDetectorConstruction()->GetWorldVolume()));
  if (vol) ans->push_back( GateVolumeSelector(vol)); else return ans;
   
  for (size_t i=1;i<numList.size();++i){
    if (comp->GetChildNumber()<1) break;
    G4int num = numList[i];
    if (num>=0){
      size_t numChild=0;
      while (num >= (G4int)comp->GetChildComponent(numChild)->GetVolumeNumber()){
        num -= comp->GetChildComponent(numChild)->GetVolumeNumber();
        if (numChild < comp->GetChildNumber()-1)
          numChild++;
        else
          break;
      }

      comp = comp->GetChildComponent(numChild);
      if (!comp || !comp->IsActive()) continue;
      vol = comp->GetPhysicalVolume(num);

      while (last_vol){
        G4LogicalVolume* logical = last_vol->GetLogicalVolume();
        if (!logical->IsDaughter(vol)){
          G4bool pb=true;
          for (unsigned int ii=0;ii<logical->GetNoDaughters();++ii){
            last_vol=logical->GetDaughter(ii);
            if (last_vol->GetLogicalVolume()->IsAncestor(vol)) {
              ans->push_back(last_vol);
              pb=false;
              break;
            }
          }
          if (pb) return ans; // no last_vol child is ancestor of vol...
        } else {
          ans->push_back(vol);
          last_vol=vol;
          break;
        }
      }
    } else {
      if (comp->GetChildNumber() != 1) break;
      comp = comp->GetChildComponent(0);
      if (!comp || comp->IsActive()) break;
    }
  }
  return ans;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateVSystem::ComputeObjectCenter(const GateVolumeID* volID) const
{

  G4RotationMatrix rotation ;
  G4ThreeVector translation ;
  G4VPhysicalVolume* vol=0;
  for (size_t i=0;i<volID->size();i++){
    vol = volID->GetVolume(i);
    G4RotationMatrix rot= vol->GetObjectRotationValue() ;
    G4ThreeVector transl= vol->GetObjectTranslation() ;

    translation = translation + rotation*transl;
    rotation = rotation * rot;
  }
  if (!vol) return G4ThreeVector();
  G4VSolid* solid = vol->GetLogicalVolume()->GetSolid();
  const G4VoxelLimits noLimits;
  G4double x[2],y[2],z[2];
  G4bool ok=true;
   
  G4AffineTransform transform(rotation,translation);
  ok = ok && solid->CalculateExtent(kXAxis,noLimits,transform,x[0],x[1]) ;
  ok = ok && solid->CalculateExtent(kYAxis,noLimits,transform,y[0],y[1]) ;
  ok = ok && solid->CalculateExtent(kZAxis,noLimits,transform,z[0],z[1]) ;

  return ok ? G4ThreeVector  ((x[0]+x[1])/2
                              ,(y[0]+y[1])/2
			      ,(z[0]+z[1])/2)
    :
    G4ThreeVector(0,0,0);
}
//-----------------------------------------------------------------------------


//OK GND 2022 TODO change name when possible
G4int GateVSystem::GetMainComponentIDGND(const GateDigi& digi)
   {  return digi.GetComponentID(m_mainComponentDepth) ; }



