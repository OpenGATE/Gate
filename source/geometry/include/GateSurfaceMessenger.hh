/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateSurfaceMessenger_hh
#define GateSurfaceMessenger_hh

#include "GateClockDependentMessenger.hh"
#include "GateSurface.hh"
#include "G4UIcmdWithAString.hh"

class G4UIcommand;

/*! \class GateSurfaceMessenger
  
  \brief The messenger belonging to the GateSurface. 

  - Add one macro command, namely one to specify the name of the optical surface. The properties
    of the optical surface are read from Surfaces.xml using this name.
  
  - GateSurfaceMessenger - by d.j.vanderlaan@tnw.tudelft.nl
  
  */
class GateSurfaceMessenger : public GateClockDependentMessenger
{
  public:
    //! constructor
    GateSurfaceMessenger(GateSurface* itsSurface);
    //! destructor
    virtual ~GateSurfaceMessenger();

    //! returns the GateSurface this messenger belongs to
    GateSurface* GetSurface()
    { return (GateSurface*)GetClockDependent();}

    //! processes the macro commands
    virtual void SetNewValue(G4UIcommand*, G4String newValue);

  private:
    G4UIcmdWithAString* m_setSurfaceCmd; //!< command to set the optical surface name
};

#endif

#endif
