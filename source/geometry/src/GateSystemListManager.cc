/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
   Revision v6.2   2012/07/09  by vesna.cuplov@gmail.com
   Added the new OpticalSystem for optical imaging.
*/

#include "GateSystemListManager.hh"
#include "GateSystemListMessenger.hh"
#include "GateScannerSystem.hh"
#include "GateCylindricalPETSystem.hh"
#include "GateSPECTHeadSystem.hh"
#include "GateOpticalSystem.hh" // v. cuplov
#include "GateEcatSystem.hh"
#include "GateEcatAccelSystem.hh"
#include "GateCPETSystem.hh"
#include "GatePETScannerSystem.hh"
#include "GateCTScannerSystem.hh"
#include "GateOPETSystem.hh"
#include "GateSystemComponent.hh"
#include "GateVVolume.hh"

//-----------------------------------------------------------------------------
// Static pointer to the GateSystemListManager singleton
GateSystemListManager* GateSystemListManager::theGateSystemListManager=0;

//-----------------------------------------------------------------------------
// The list of predefined-system names
// Note that this list must always be terminated with an empty chain
const G4String GateSystemListManager::theSystemNameList[]= {
  "cylindrical1",
  "cylindricalPET",
  "SPECThead",
  "scanner",
  "ecat",
  "ecatAccel",
  "CPET",
  "PETscanner",
  "CTscanner",
  "OPET",
  "OpticalSystem", // Optical System name - v. cuplov
  ""
};
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/*    	This function allows to retrieve the current instance of the GateSystemListManager singleton
      	If the GateSystemListManager already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateSystemListManager constructor
*/
GateSystemListManager* GateSystemListManager::GetInstance()
{
  if (!theGateSystemListManager)
    theGateSystemListManager = new GateSystemListManager();
  return theGateSystemListManager;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
// Private constructor
GateSystemListManager::GateSystemListManager()
  : GateListManager( "systems", "system", false, false )
{
	m_isAnySystemDefined=true;
  m_messenger = new GateSystemListMessenger(this);
  theInsertedSystemsNames = new std::vector<G4String>;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Public destructor
GateSystemListManager::~GateSystemListManager()
{  
  delete m_messenger;
  delete theInsertedSystemsNames;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Registers a new object-system in the system list
void GateSystemListManager::RegisterSystem(GateVSystem* newSystem)
{   
  theListOfNamedObject.push_back(newSystem);  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Removes a deleted object-system from the system-list    
void GateSystemListManager::UnregisterSystem(GateVSystem* aSystem) 
{
  theListOfNamedObject.erase( std::remove(theListOfNamedObject.begin(), theListOfNamedObject.end(), aSystem ) );
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/* Tries to find to which system an inserter is attached
   (either directly or through one of its ancestors)
   anCreator: the inserter to test  
   Returns: the system to which the inserter is attached, if any
*/
GateVSystem* GateSystemListManager::FindSystemOfCreator(GateVVolume* anCreator)
{
  
  for (GateListOfNamedObject::iterator iter = theListOfNamedObject.begin(); iter!=theListOfNamedObject.end(); ++iter)
    if ( ((GateVSystem*)(*iter))->CheckConnectionToCreator(anCreator) )
      return ((GateVSystem*)(*iter));
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/* this function underwent an important modification for the multi-system approach.
   Check whether the name of a new inserter,or the system type, has the same name as one of the predefined systems
   If that's the case, auto-create the system
   newChildCreator: the newly-created inserter  
*/
void GateSystemListManager::CheckScannerAutoCreation(GateVVolume* newChildCreator)
{
  // Check whether the name of the new inserter,or the system type, has the same name as one of the predefined systems
   G4String baseName;
  
   if(m_messenger->GetSystemType().empty() == false)
   {
      baseName = m_messenger->GetSystemType();
      m_messenger->SetSystemType("");
   }
   else baseName= newChildCreator->GetObjectName();

   if (DecodeTypeName(baseName)>=0) { 
    // A predefined-system name was recognised: launch the autocreation of the system
      GateMessage("Core", 2, "[GateSystemListManager::CheckScannerAutoCreation:\n"
            << "\tCreating new system based on volume inserter '" << baseName << "'\n");

    // Create the system
      theInsertedSystemsNames->push_back(newChildCreator->GetObjectName());
      GateVSystem* newSystem = InsertNewSystem(baseName);
    
    // Attach the system's base to the inserter
      newSystem->GetBaseComponent()->SetCreator(newChildCreator);        
   }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/* Checks whether a name corresponds to onw of the predefined system-names    
   name: the name to check	
   returns:  the position of the name in the name-table (-1 if not found)
*/
G4int GateSystemListManager::DecodeTypeName(const G4String& name)
{
  size_t i;
  for (i=0; !( theSystemNameList[i].empty() ) ; i++)
    if (theSystemNameList[i]==name)
      return i;
  return -1;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/* Create a new system of a specific type      	
   typeName: the type-name of the system to create	
   returns the newly created system
*/
GateVSystem* GateSystemListManager::InsertNewSystem(const G4String& typeName)
{
  
  // Finds the type-name in the type-name table
  G4int typePos = DecodeTypeName(typeName);

  // Create the requested system
  GateVSystem* newSystem=0;
  switch (typePos) 
    {
    case 0:
    case 1:
      newSystem = new GateCylindricalPETSystem(MakeElementName(typeName));
      break;
    case 2:
      newSystem = new GateSPECTHeadSystem(MakeElementName(typeName));
      break;
    case 3:
      newSystem = new GateScannerSystem(MakeElementName(typeName));
      break;
    case 4:
      newSystem = new GateEcatSystem(MakeElementName(typeName));
      break;
    case 5:
      newSystem = new GateEcatAccelSystem(MakeElementName(typeName));
      break;
    case 6:
      newSystem = new GateCPETSystem(MakeElementName(typeName));
      break;
    case 7:
      newSystem = new GatePETScannerSystem(MakeElementName(typeName));
      break;
    case 8:
      newSystem = new GateCTScannerSystem(MakeElementName(typeName));
      break;
    case 9:  
      newSystem = new GateOPETSystem(MakeElementName(typeName));
      break;
    case 10:  
      newSystem = new GateOpticalSystem(MakeElementName(typeName));  // Optical System - v. cuplov
      break;
    default:
      G4cout << "System type name '" << typeName << "' was not recognised --> insertion request must be ignored!\n";
      break;
    }
  
  return newSystem;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Lists all the system-names into a string
const G4String& GateSystemListManager::DumpChoices()
{
  static G4String theList;
  size_t i;
  if (theList.empty()) {
    for (i=0; !( theSystemNameList[i].empty() ) ; i++)
      theList += theSystemNameList[i] + " ";
  }
  return theList;
}
//-----------------------------------------------------------------------------



