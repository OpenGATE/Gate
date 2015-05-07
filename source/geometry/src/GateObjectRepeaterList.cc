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
  GateMessage("Repeater", 8, itsName << " itsElementTypeName = " << itsElementTypeName << Gateendl;);
     
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
  for (iterator it=theListOfNamedObject.begin(); it!=theListOfNamedObject.end(); it++) 
    ((GateVGlobalPlacement*)(*it))->Describe(indent+1);
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
    for (iterator it=theListOfNamedObject.begin(); it!=theListOfNamedObject.end(); it++)
      pQueue = ((GateVGlobalPlacement*)(*it))->ComputePlacements(pQueue);

  return pQueue;
} 
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateObjectRepeaterList::ComputeParameters()
{
  G4double aTime = GetCurrentTime();
  for (iterator it=theListOfNamedObject.begin(); it!=theListOfNamedObject.end(); it++)
    if ( ((GateVGlobalPlacement*)(*it)) ) 
      ((GateVGlobalPlacement*)(*it))->ComputeParameters(aTime);
}
//-------------------------------------------------------------------------------------

