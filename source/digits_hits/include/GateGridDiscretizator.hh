/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDiscretization.cc for more detals
  \sa GateDiscretization, GateDiscretizationMessenger
  // OK GND 2022
  
  \class  GridDiscretizator
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/



#ifndef GateGridDiscretizator_h
#define GateGridDiscretizator_h 1

#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateGridDiscretizatorMessenger.hh"
#include "GateSinglesDigitizer.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"

#include "GateMaps.hh"
#include <set>

class GateGridDiscretizatorMessenger;

class GateGridDiscretizator : public GateVDigitizerModule
{
public:
  
  GateGridDiscretizator(GateSinglesDigitizer *digitizer, G4String name);
  ~GateGridDiscretizator();
  
  //G4int ChooseVolume(G4String val);

  void Digitize() override;

  void SetDiscretization(G4double val)   { m_GateGridDiscretizator = val; };


  void SetStripOffsetX( G4double val)   {  stripOffsetX = val;  };
  void SetStripOffsetY( G4double val)   {  stripOffsetY = val;  };
  void SetStripOffsetZ( G4double val)   {  stripOffsetZ = val;  };
 
  void SetNumberStripsX(int val)   {  numberStripsX = val;  };
  void SetNumberStripsY( int val)   {  numberStripsY = val;  };
  void SetNumberStripsZ( int val)   {  numberStripsZ = val;  };
 
  void SetStripWidthX(G4double val)   {  stripWidthX = val;  };
  void SetStripWidthY( G4double val)   {  stripWidthY = val;  };
  void SetStripWidthZ( G4double val)   {  stripWidthZ = val;  };
  
  void SetNumberReadOutBlocksX(int val)   {  numberReadOutBlocksX = val;  };
  void SetNumberReadOutBlocksY( int val)   {  numberReadOutBlocksY = val;  };
  void SetNumberReadOutBlocksZ( int val)   {  numberReadOutBlocksZ = val;  };


protected:


  //void SetVolumeName(G4String name) {
  //	  m_name=name;};

  void DescribeMyself(size_t );

  void SetGridPoints3D( int indexX, int indexY, int indexZ, G4ThreeVector& pos );


  int GetXIndex(G4double posX);
  int GetYIndex(G4double posY);
  int GetZIndex(G4double posZ);

  void ApplyBlockReadOut();

protected:
  G4double   m_GateGridDiscretizator;

  G4double stripOffsetX;
  G4double stripOffsetY;
  G4double stripOffsetZ;
  G4double stripWidthX;
  G4double stripWidthY;
  G4double stripWidthZ;
  G4int numberStripsX;
  G4int numberStripsY;
  G4int numberStripsZ;
  G4int numberReadOutBlocksX;
  G4int numberReadOutBlocksY;
  G4int numberReadOutBlocksZ;
  G4ThreeVector volSize;
  G4double deadSpX;
  G4double deadSpY;
  G4double deadSpZ;
  G4double pitchY;
  G4double pitchX;
  G4double pitchZ;


std::vector<int > index_X_list;
std::vector<int > index_Y_list;
std::vector<int > index_Z_list;

std::map<std::tuple<int,int,int>, std::vector<int>> blockIndex;


private:
  GateDigi* m_outputDigi;

  GateGridDiscretizatorMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  G4String m_name;

  G4VoxelLimits limits;
  G4double min, max;
  G4AffineTransform at;

  int INVALID_INDEX=-2;
  double EPSILON=1e-9;


};

#endif








