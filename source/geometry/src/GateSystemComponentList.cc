/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSystemComponentList.hh"

#include "GateTools.hh"
#include "GateSystemComponent.hh"
#include "GateSystemComponentListMessenger.hh"



GateSystemComponentList::GateSystemComponentList(GateSystemComponent* itsMother,
						 const G4String& itsName)
  : GateModuleListManager(itsMother,itsName,"daughter")
{
//    G4cout << " DEBUT GateSystemComponentList"  << G4endl;
    m_messenger = new GateSystemComponentListMessenger(this);
//    G4cout << " FIN GateSystemComponentList"  << G4endl;
}




GateSystemComponentList::~GateSystemComponentList()
{  
    delete m_messenger;
}




void GateSystemComponentList::InsertChildComponent(GateSystemComponent* newChildComponent)
{
  theListOfNamedObject.push_back(newChildComponent);
}



void GateSystemComponentList::DescribeChildComponents(size_t indent,G4bool recursiveDescribe)
{
  G4cout << GateTools::Indent(indent) << "Nb of children:       " << theListOfNamedObject.size() << "\n";
  for (size_t i=0; i<theListOfNamedObject.size(); i++)
    if (recursiveDescribe)
      GetChildComponent(i)->Describe(indent+1);
    else
      G4cout << GateTools::Indent(indent+1) << GetChildComponent(i)->GetObjectName() << G4endl;
}

void GateSystemComponentList::ListElements()
{
  DescribeChildComponents(0);
}


/* Check whether an inserter is connected to the component tree
      	
	anCreator: the inserter we want to check
	
	returns true if the inserter is attached to one of the components
*/
G4bool GateSystemComponentList::CheckConnectionToCreator(GateVVolume* anCreator) 
{ 
  for ( size_t i=0 ; i < GetChildNumber() ; ++i) 
    if ( GetChildComponent(i)->CheckConnectionToCreator(anCreator) )
      return true;

  return false;
}
    
    
    

// Compute the maximum depth of the child component trees
size_t GateSystemComponentList::GetMaxChildTreeDepth()
{
  // Compute the maximum distance between the component and its descendants
  // We ask each child to compute its own tree-depth, and We store the maximum of the child tree depths
  size_t childTreeDepth=0;
  for ( size_t i=0 ; i<GetChildNumber() ; ++i) {
    size_t childResult = GetChildComponent(i)->GetTreeDepth();
    if (childResult>childTreeDepth)
      childTreeDepth=childResult;
  }
  
  return childTreeDepth;
}



// Compute the number of active daughter-components (i.e. components that are linked to an inserter)
size_t GateSystemComponentList::GetActiveChildNumber()
{
  // We loop on the list of daughter-components
  // We increment the number of active children for each daughter-component that is active
  size_t activeChildNumber=0;
  for ( size_t i=0 ; i<GetChildNumber() ; ++i) 
    if ( GetChildComponent(i)->IsActive() )
      ++activeChildNumber;
  
  return activeChildNumber;
}



// Finds a child component from its name
GateSystemComponent* GateSystemComponentList::FindSystemComponent(const G4String& componentName)
{
  for ( size_t i=0 ; i<GetChildNumber() ; ++i) {
      GateSystemComponent* childResult = GetChildComponent(i)->FindSystemComponent(componentName);
      if (childResult)
      	return childResult;
  }

  // No match was found: return a null pointer
  return 0;
}
