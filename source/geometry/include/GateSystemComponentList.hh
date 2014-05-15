/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSystemComponentList_h
#define GateSystemComponentList_h 1

#include "globals.hh"

#include "GateModuleListManager.hh"

class GateSystemComponent;
class GateVVolume;
class GateSystemComponentListMessenger;

class GateSystemComponentList : public GateModuleListManager
{
  public:
    GateSystemComponentList(GateSystemComponent* itsMother,
    			    const G4String& itsName);
    virtual ~GateSystemComponentList();

     virtual void InsertChildComponent(GateSystemComponent* newChildComponent);

     virtual void DescribeChildComponents(size_t indent,G4bool recursiveDescribe=false);
     virtual void ListElements();
     virtual GateSystemComponent* FindChildComponent(const G4String& name)
      	  { return (GateSystemComponent*) FindElement(name); }
     virtual GateSystemComponent* GetChildComponent(size_t i)
      	  {return (GateSystemComponent*) GetElement(i);}
     inline GateSystemComponent* GetMotherComponent()
      	  { return (GateSystemComponent*) GetMotherObject() ;}
     virtual size_t GetChildNumber()
      	  { return size();}

    /*! \brief Check whether an inserter is connected to the component tree
      	
	\param anCreator: the inserter we want to check
	
	\return true if the inserter is attached to one of the componentsS
    */
    G4bool CheckConnectionToCreator(GateVVolume* anCreator);
    
    //! Compute the number of active daughter-components (i.e. components that are linked to an inserter)
    size_t GetActiveChildNumber();

    //! Compute the maximum depth of the child component trees
    size_t GetMaxChildTreeDepth();

    //! Finds a component from its name in the component tree
    GateSystemComponent* FindSystemComponent(const G4String& componentName);

  protected:
      GateSystemComponentListMessenger*    m_messenger;
};

#endif

