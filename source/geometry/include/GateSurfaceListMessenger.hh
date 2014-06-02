/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateSurfaceListMessenger_hh
#define GateSurfaceListMessenger_hh 1

#include "globals.hh"
#include "GateListMessenger.hh"
#include "GateSurfaceList.hh"

/*! \class GateSurfaceListMessenger
  
  \brief the messenger belonging to a GateSurfaceList

  Does not define any new methods and commands. All of them are inherited 
  from GateListMessenger.
  
  - GateSurfaceListMessenger - by d.j.vanderlaan@tnw.tudelft.nl
  
  */
class GateVVolume;

class GateSurfaceListMessenger: public GateListMessenger
{
  public:
    GateSurfaceListMessenger(GateSurfaceList* itsChildList);
   ~GateSurfaceListMessenger();
    
  private:
    //! returns ""
    virtual const G4String& DumpMap();
    //! creates a GateSurface and adds it to the GateSurfaceList
    virtual void DoInsertion(const G4String& surfaceName);
    //! lists the possible choices for the insert command (inherited)
    virtual void ListChoices() ;
    //! checks if name is ok
    virtual G4bool CheckNameConflict(const G4String& name);
    //! returns the SurfaceList this messenger belongs to
    virtual GateSurfaceList* GetSurfaceList()
    { return (GateSurfaceList*) GetListManager(); }
    //! returns the GateVVolume where the list this messenger belongs to belongs to
    virtual GateVVolume* GetCreator()
    { return GetSurfaceList()->GetCreator();}

  private:
  
};

#endif

#endif
