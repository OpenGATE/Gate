/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDiscretizerModule.cc for more detals
  */


/*! \class  GateDiscretizerModule
    \brief  GateDiscretizerModule does some dummy things with input digi
    to create output digi

    - GateDiscretizerModule - by marc.granado@universite-paris-saclay.fr

    \sa GateDiscretizerModule, GateDiscretizerModuleMessenger
*/

#ifndef GateDiscretizerModule_h
#define GateDiscretizerModule_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateDiscretizerModuleMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateDiscretizerModule : public GateVDigitizerModule
{
public:
  
  GateDiscretizerModule(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDiscretizerModule();
  
  void Digitize() override;
  
  //! These functions return the resolution in use.
  	 G4String GetNameAxis()				   { return m_nameAxis;     }
  	 G4double GetResolution()   	       { return m_resolution;  }
     G4double GetResolutionX()   	       { return m_resolutionX; }
     G4double GetResolutionY()   	       { return m_resolutionY; }
     G4double GetResolutionZ()   	       { return m_resolutionZ; }

     G4double calculatePitch(G4double crystal_size, G4double spatial_resolution);


     //Set Parameter
     void SetNameAxis(const G4String&);



     //Set Variables
     void SetResolution(G4double  val)   { m_resolution  = val;  }
     void SetResolutionX(G4double val)   { m_resolutionX = val;  }
     void SetResolutionY(G4double val)   { m_resolutionY = val;  }
     void SetResolutionZ(G4double val)   { m_resolutionZ = val;  }


     void SetVirtualIDs(int nBinsX, int nBinsY,int nBinsZ,double pitchX,double pitchY,double pitchZ, G4ThreeVector& pos);
     void SetVirtualID(int nBins,double pitch, G4double pos, int depth);
     void DescribeMyself(size_t );

protected:
  // *******implement your parameters here
     G4double m_resolution;

     G4String m_nameAxis;
     G4double m_resolutionX;
     G4double m_resolutionY;
     G4double m_resolutionZ;


     //These are in Spatial resolution but aren't there rom touchable?
     //We'll need to check

     //G4Navigator* m_Navigator;
     //G4TouchableHistoryHandle m_Touchable;

private:

  G4int m_systemDepth;

  GateDigi* m_outputDigi;

  GateDiscretizerModuleMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








