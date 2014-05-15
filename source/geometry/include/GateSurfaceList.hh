/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateSurfaceList_hh
#define GateSurfaceList_hh 1

#include "globals.hh"
#include "GateModuleListManager.hh"
#include "GateSurface.hh"

class GateVVolume;
class GateSurfaceListMessenger;

/*! \class GateSurfaceList

  \brief Contains all the GateSurface object belonging to one GateVVolume
  
  Each volume in Gate has a list of surfaces associated with it. After the geometry is
  built, all surfaces are built. The BuildSurfaces() method is used for this. 
  
  - GateSurfaceList - by d.j.vanderlaan@tnw.tudelft.nl
  
  */
class GateSurfaceList : public GateModuleListManager
{
  public:
    //! constructor
    GateSurfaceList(GateVVolume* itsCreator, G4bool acceptsNewChildren);
    //! destructor
    virtual ~GateSurfaceList();

    //! class each of the surfaces to build themselves
    void BuildSurfaces();
    //! adds a surface to the list
    void AddSurface(GateSurface* surface);
    //! displays a list of all the surfaces
    void DescribeSurfaces(size_t indent=0);
     
    //! Calls DescribeSurfaces(0)
    void ListElements();

    //! Returns the surface with the specified name
    GateSurface*        FindSurface(const G4String& name) { return (GateSurface*) FindElement(name); }
    //! Returns the ith surface in the list
    GateSurface*        GetSurface(size_t i) {return (GateSurface*) GetElement(i);}
    //! Return the GateVVolume this list belongs to
    GateVVolume* GetCreator() const { return (GateVVolume*) GetMotherObject() ;}

  protected:
     GateSurfaceListMessenger*    m_messenger;  //!< its messenger
};

#endif

#endif
