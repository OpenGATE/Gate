/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateVirtualSegmentationSD.cc for more detals
  */


/*! \class  GateVirtualSegmentationSD
    \brief  GateVirtualSegmentationSD  - For each volume the local position of the interactions within the crystal are virtually segmented.
	the X,Y,Z vector is translated into the IDs of virtual divisions within the crystal


    - GateVirtualSegmentationSD - by marc.granado@universite-paris-saclay.fr

    \sa GateVirtualSegmentationSD, GateVirtualSegmentationSDMessenger
*/

#ifndef GateVirtualSegmentationSD_h
#define GateVirtualSegmentationSD_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateVirtualSegmentationSDMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateVirtualSegmentationSD : public GateVDigitizerModule
{
public:
  
  GateVirtualSegmentationSD(GateSinglesDigitizer *digitizer, G4String name);
  
 ~GateVirtualSegmentationSD();

  void Digitize() override;
  
  //! These functions return the pitch in use.
  	 G4String GetNameAxis()				   { return m_nameAxis;     }
  	 G4double GetPitch()   	       { return m_pitch;  }
     G4double GetPitchX()   	       { return m_pitchX; }
     G4double GetPitchY()   	       { return m_pitchY; }
     G4double GetPitchZ()   	       { return m_pitchZ; }

     G4double calculatePitch(G4double crystal_size, G4double target_pitch);


     //Set Parameter
     void SetNameAxis(const G4String&);

     void SetUseMacroGenerator(G4bool val) {useMacroGenerator=val;}

     //Set Variables
     void SetPitch(G4double  val)   { m_pitch  = val;  }
     void SetPitchX(G4double val)   { m_pitchX = val;  }
     void SetPitchY(G4double val)   { m_pitchY = val;  }
     void SetPitchZ(G4double val)   { m_pitchZ = val;  }

     void SetParameters();

     void SetVirtualID(int nBins,double pitch, G4double pos, int depth);
     void DescribeMyself(size_t );

protected:
  // *******implement your parameters here
     G4double m_pitch;

     G4String m_nameAxis;
     G4double m_pitchX;
     G4double m_pitchY;
     G4double m_pitchZ;

 	G4int nBinsX;
 	G4int nBinsY;
 	G4int nBinsZ;

 	G4double pitchX;
 	G4double pitchY;
 	G4double pitchZ;

 	G4int depthX;
 	G4int depthY;
 	G4int depthZ;

 	G4double xLength;
 	G4double yLength;
 	G4double zLength;

     //These are in Spatial pitch but aren't there rom touchable?
     //We'll need to check

     //G4Navigator* m_Navigator;
     //G4TouchableHistoryHandle m_Touchable;

private:

  G4int m_systemDepth;

  G4int m_IsFirstEntry;

  G4bool useMacroGenerator;

  GateDigi* m_outputDigi;

  GateVirtualSegmentationSDMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};


#endif








