/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateObjectRepeaterList_h
#define GateObjectRepeaterList_h 1

#include "globals.hh"

#include "GateModuleListManager.hh"
#include "GateVGlobalPlacement.hh"
#include "GateVVolume.hh"

//class GateVObjectMove;
//class GateVolumePlacement;
class GateListMessenger;

class GateObjectRepeaterList : public GateModuleListManager
{
public:
  GateObjectRepeaterList(GateVVolume* itsInserter,
                         const G4String& itsName,
                         const G4String& itsElementTypeName);
  virtual ~GateObjectRepeaterList();

public:
  virtual GatePlacementQueue* ComputePlacements(GatePlacementQueue* pQueue);
  virtual void AppendObjectRepeater(GateVGlobalPlacement* newObjectRepeater);
  virtual void DescribeRepeaters(size_t indent=0);

  virtual void ListElements();
  virtual GateVGlobalPlacement* FindInserter(const G4String& name)
  { return (GateVGlobalPlacement*) FindElement(name); }
  inline GateVGlobalPlacement* GetRepeater(size_t i)
  {return (GateVGlobalPlacement*) GetElement(i);}
  inline GateVVolume* GetCreator()
  { return (GateVVolume*) GetMotherObject() ;}
	  
  virtual void ComputeParameters();

  virtual inline G4String MakeElementName(const G4String& newBaseName)
  { return GetCreator()->GetObjectName() + "/" + newBaseName; }


   
protected:
  GateListMessenger*   m_messenger;
    
  G4String m_typeName;
};

#endif

