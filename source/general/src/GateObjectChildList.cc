/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateObjectChildList.hh"
#include "GateObjectChildListMessenger.hh"

#include "GateVVolume.hh"
#include "GateTools.hh"

class G4LogicalVolume;
class G4Material;

//-----------------------------------------------------------------------------------------------------------------
GateObjectChildList::GateObjectChildList(GateVVolume* itsCreator, G4bool acceptNewChildren) :
  GateModuleListManager(itsCreator,itsCreator->GetObjectName()+"/daughters", "daughter", false, acceptNewChildren),
  pMessenger(0)
{ 
  if (acceptNewChildren) pMessenger = new GateObjectChildListMessenger(this);
 
}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
GateObjectChildList::~GateObjectChildList()
{ 
  if (pMessenger) delete pMessenger;
//GetPhysicalVolumeName()
  /*for (size_t i=0; i<theListOfNamedObject.size(); i++){
      if (theListOfNamedObject[i])        GetVolume(i)->DestroyGeometry();  */
}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
void GateObjectChildList::ConstructChildGeometry(G4LogicalVolume* logical, G4bool flagUpdateOnly)
{ 
 
  for (iterator it=begin(); it!=end(); it++)
  { 
    if (*it) ((GateVVolume*)(*it))->ConstructGeometry(logical, flagUpdateOnly);
  }

}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
void GateObjectChildList::DestroyChildGeometry()
{
  
  if (IsEnabled()){
    for (iterator it=begin(); it!=end(); it++)
    {
      if (*it) ((GateVVolume*)(*it))->DestroyGeometry();  
	/*G4cout<<"GateObjectChildList :: listObject = "<<i<< Gateendl;*/}
   }
   		
}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
void GateObjectChildList::AddChild(GateVVolume* pnewChildCreator)
{


  theListOfNamedObject.push_back(pnewChildCreator);
  
  pnewChildCreator->SetMotherList(this);
  
}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
void GateObjectChildList::DescribeChildren(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Nb of children:        " << size() << Gateendl;
  for (iterator it=begin(); it!=end(); it++)
    if (*it) 
    	  G4cout << GateTools::Indent(indent+1) << "child: '" << ((GateVVolume*)(*it))->GetObjectName() << "'\n";
    else
    	  G4cout << GateTools::Indent(indent+1) << "detached child\n";
}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
void GateObjectChildList::ListElements()
{ DescribeChildren(0);}
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
G4int GateObjectChildList::GetChildNo(GateVVolume* anInserter, G4int copyNo)
{
  G4int childNo=0;
  
  for (iterator it=begin(); it!=end(); it++)
  {
    if ( ((GateVVolume*)(*it)) != anInserter ) 
      childNo += ((GateVVolume*)(*it))->GetVolumeNumber();
    else
      return childNo+copyNo;
  }

  G4cout << "[" << GetCreator()->GetObjectName() << "]: could not find any entry for inserter '" << anInserter->GetObjectName() << "'" <<  Gateendl;
  return -1;
}
//-----------------------------------------------------------------------------------------------------------------
