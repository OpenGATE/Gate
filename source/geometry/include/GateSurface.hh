/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateSurface_hh
#define GateSurface_hh

#include <vector>

#include "globals.hh"
#include "GateClockDependent.hh"
#include "GateVVolume.hh"

class GateSurfaceMessenger;
class G4LogicalBorderSurface;
class G4OpticalSurface;

/*! \class GateSurface
  
  \brief Is a surface between two GateVVolumes. 

  A surface is defined by specifying two volumes, or GateVVolumes. The first one has 
  to be speciefied in the constructor, the second by one of the methods. 

  - Has methods to load the surface properties from an xml-file
  
  - GateSurface - by d.j.vanderlaan@tnw.tudelft.nl
  
  */
class GateSurface : public GateClockDependent
{
  public:
    //! constructor
    /**
      \param itsName: the name of the surface
      \param inserter: the first GateVObjectInsterter that defines the surface
      */
    GateSurface(const G4String& itsName, GateVVolume* inserter);
    //! destructor
    virtual ~GateSurface();

    //! returns the first GateVVolume
    inline GateVVolume* GetInserter1() const
    { return m_inserter1;}

    //! sets the second GateVVolume btween which the surface exists
    void SetInserter2(GateVVolume* inserter);
    //! returns the first GateVVolume
    inline GateVVolume* GetInserter2() const
    { return m_inserter2;}

    //! set the name of the surfaceproperties that are to be read from the xml-file
    void SetOpticalSurfaceName(const G4String& name);
    //! returns the name of the surface properties
    inline const std::string& GetOpticalSurfaceName() const
    { return m_opticalsurfacename;}

    //! creates the surfaces 
    /** Creates the Geant4 objects that are needed to define a surface */
    void BuildSurfaces();

  private:
    //! deletes the Geant4 objects
    void DeleteSurfaces();
    //! reads the surface properties from the Surfaces.xml file
    /** \param name specifies the surface that has to read from the xml-file */
    G4OpticalSurface* ReadOpticalSurface(const G4String& name) const;
    
  private:
    G4String                             m_opticalsurfacename; //!< Name of the surface properties, to be read from the xml-file
    GateVVolume*                 m_inserter1;          //!< first volume
    GateVVolume*                 m_inserter2;          //!< second volume
    std::vector<G4LogicalBorderSurface*> m_surfaces;           //!< Vector containing all the Geant4 surfaces that exist bewteen the two GateVVolumes
    G4OpticalSurface*                    m_opticalsurface;     //!< The optical surface: contains the optical properties of the surface
    GateSurfaceMessenger*                m_messenger;          //!< the messenger
};

#endif

#endif
