/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateObjectChildList_hh
#define GateObjectChildList_hh 1

#include "GateModuleListManager.hh"

#include "globals.hh"

class GateObjectChildListMessenger;
class GateVVolume;
class G4LogicalVolume;
class G4Material;


/*! \class GateObjectChildList

  \brief 
  
  */
class GateObjectChildList : public GateModuleListManager
{

public:
  //! constructor
  GateObjectChildList(GateVVolume* itsCreator, G4bool acceptsNewChildren=true);
  //! destructor
  virtual ~GateObjectChildList();
  
  // Construct the child geometry
  virtual void ConstructChildGeometry(G4LogicalVolume*, G4bool);
  
  // Destroy the geometry of chldren
  virtual void DestroyChildGeometry();
  
  // Insert a new object child  
  virtual void AddChild(GateVVolume* newChildCreator);
  virtual void DescribeChildren(size_t indent=0);
     
//  virtual void ComputeParameters();
    
     
  //! 
  virtual void ListElements();

  //! Returns the volume with the specified name
  virtual GateVVolume*       FindVolume(const G4String& name) { return (GateVVolume*) FindElement(name); }
  
  //! Returns the ith volume in the list
  virtual GateVVolume*       GetVolume(size_t i) {return (GateVVolume*) GetElement(i);}
  
  //! 
  virtual GateVVolume*       GetCreator() const { return (GateVVolume*) GetMotherObject() ;}

  //! 
  virtual G4int GetChildNo(GateVVolume* anInserter, G4int copyNo);
  
protected:
  GateObjectChildListMessenger*    pMessenger;  //!< its messenger

};

#endif
