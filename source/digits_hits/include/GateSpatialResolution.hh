/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateSpatialResolution
    \brief  GateSpatialResolution does some dummy things with input digi
    to create output digi

    \sa GateSpatialResolution, GateSpatialResolutionMessenger
*/

#ifndef GateSpatialResolution_h
#define GateSpatialResolution_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateSpatialResolutionMessenger.hh"

#include "G4VoxelLimits.hh"
#include "G4TouchableHistoryHandle.hh"
#include "GateSinglesDigitizer.hh"

class GateSpatialResolution : public GateVDigitizerModule
{
public:
  
  GateSpatialResolution(GateSinglesDigitizer *digitizer, G4String name);
  ~GateSpatialResolution();
  
  void Digitize() override;

  //! These functions return the resolution in use.
    G4double GetFWHM()   	       { return m_fwhm; }
    G4double GetFWHMx()   	       { return m_fwhmX; }
    G4double GetFWHMy()   	       { return m_fwhmY; }
    G4double GetFWHMz()   	       { return m_fwhmZ; }

    //! These functions set the spresolution of a gaussian spblurring.
    /*!
      If you want a resolution of 10%, SetSpresolution(0.1)
    */
    void SetFWHM(G4double val)   { m_fwhm = val;  }
    void SetFWHMx(G4double val)   { m_fwhmX = val;  }
    void SetFWHMy(G4double val)   { m_fwhmY = val;  }
    void SetFWHMz(G4double val)   { m_fwhmZ = val;  }


    inline void ConfineInsideOfSmallestElement(const G4bool& value) { m_IsConfined = value; };
    inline G4bool IsConfinedInsideOfSmallestElement() const  	      	{ return m_IsConfined; }

    void UpdatePos(G4double ,G4double ,G4double );
    void LocateOutputDigi(GateDigi* inputDigi, G4double PxNew,G4double PyNew,G4double PzNew);

    void UpdateVolumeID();


    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the blurring
    void DescribeMyself(size_t );

protected:
    G4double m_fwhm;

    G4double m_fwhmX;
    G4double m_fwhmY;
    G4double m_fwhmZ;
    G4bool m_IsConfined;
    G4Navigator* m_Navigator;
    G4TouchableHistoryHandle m_Touchable;



private:

    G4int m_systemDepth;

  GateDigi* m_outputDigi;

  GateSpatialResolutionMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  G4VoxelLimits limits;
  G4double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
  G4AffineTransform at;
};

#endif








