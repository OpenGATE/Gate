/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateObjectRepeaterList.hh"

#include "GateDetectorConstruction.hh"
#include "GateVVolume.hh"
#include "GateVGlobalPlacement.hh"
#include "GateListManager.hh"
#include "GateObjectMoveListMessenger.hh"
#include "GateObjectRepeaterListMessenger.hh"
#include "GateTools.hh"

#include "GateMessageManager.hh"

//-------------------------------------------------------------------------------------
GateObjectRepeaterList::GateObjectRepeaterList( GateVVolume* itsInserter,
    			   			const G4String& itsName,
			   			const G4String& itsElementTypeName )
  : GateModuleListManager(itsInserter,itsName,itsElementTypeName ),
    m_messenger(0),
    m_typeName(itsElementTypeName)
{
  GateMessage("Repeater", 8, itsName << " GateObjectRepeaterList::GateObjectRepeaterList\n");
  GateMessage("Repeater", 8, itsName << " itsElementTypeName = " << itsElementTypeName << G4endl;);
     
  if (itsElementTypeName == "move" )
    {     
      m_messenger = new GateObjectMoveListMessenger(this);}
  else 
    m_messenger = new GateObjectRepeaterListMessenger(this);
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
GateObjectRepeaterList::~GateObjectRepeaterList()
{  
  if (m_messenger)
    delete m_messenger;
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateObjectRepeaterList::AppendObjectRepeater(GateVGlobalPlacement* newObjectRepeater)
{
  theListOfNamedObject.push_back(newObjectRepeater);
  ComputeParameters();
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateObjectRepeaterList::DescribeRepeaters(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Nb of " << m_typeName << "s:       " << theListOfNamedObject.size() << "\n";
  for (size_t i=0; i<theListOfNamedObject.size() ; i++)
    GetRepeater(i)->Describe(indent+1);
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateObjectRepeaterList::ListElements()
{
  DescribeRepeaters(0);
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
GatePlacementQueue* GateObjectRepeaterList::ComputePlacements(GatePlacementQueue *pQueue)
{
  if (IsEnabled())
    for (size_t i=0; i<theListOfNamedObject.size() ; i++)
      pQueue = GetRepeater(i)->ComputePlacements(pQueue);

  return pQueue;
} 
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateObjectRepeaterList::ComputeParameters()
{
  G4double aTime = GetCurrentTime();
  for (size_t i=0; i<theListOfNamedObject.size(); i++)
    if ( GetRepeater(i) ) 
      GetRepeater(i)->ComputeParameters(aTime);
}
//-------------------------------------------------------------------------------------

