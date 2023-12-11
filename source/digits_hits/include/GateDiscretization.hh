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
  
  \class  GateDiscretization
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/



#ifndef GateDiscretization_h
#define GateDiscretization_h 1

#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateDiscretizationMessenger.hh"
#include "GateSinglesDigitizer.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"
#include <set>

class GateDiscretizationMessenger;

class GateDiscretization : public GateVDigitizerModule
{
public:
  
  GateDiscretization(GateSinglesDigitizer *digitizer, G4String name);
  ~GateDiscretization();
  
  G4int ChooseVolume(G4String val);

  void Digitize() override;

  void SetDiscretization(G4double val)   { m_GateDiscretization = val; };


  void SetStripOffsetX(G4String name, G4double val)   {  m_param.stripOffsetX = val;  };
  void SetStripOffsetY(G4String name, G4double val)   {  m_param.stripOffsetY = val;  };
  void SetStripOffsetZ(G4String name, G4double val)   {  m_param.stripOffsetZ = val;  };
 
  void SetNumberStripsX(G4String name,int val)   {  m_param.numberStripsX = val;  };
  void SetNumberStripsY(G4String name, int val)   {  m_param.numberStripsY = val;  };
  void SetNumberStripsZ(G4String name, int val)   {  m_param.numberStripsZ = val;  };
 
  void SetStripWidthX(G4String name,G4double val)   {  m_param.stripWidthX = val;  };
  void SetStripWidthY(G4String name, G4double val)   {  m_param.stripWidthY = val;  };
  void SetStripWidthZ(G4String name, G4double val)   {  m_param.stripWidthZ = val;  };
  
  void SetNumberReadOutBlocksX(G4String name,int val)   {  m_param.numberReadOutBlocksX = val;  };
  void SetNumberReadOutBlocksY(G4String name, int val)   {  m_param.numberReadOutBlocksY = val;  };
  void SetNumberReadOutBlocksZ(G4String name, int val)   {  m_param.numberReadOutBlocksZ = val;  };
  struct param {

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
  };
  param m_param;

  std::vector<int > index_X_list;
  std::vector<int > index_Y_list;
  std::vector<int > index_Z_list;

  std::map<std::tuple<int,int,int>, std::vector<int>> blockIndex;

  void SetVolumeName(G4String name) {
	  G4cout<<"seting m_name Volume "<<name<<G4endl;
	  m_name=name;};

  void DescribeMyself(size_t );

  void SetGridPoints3D( int indexX, int indexY, int indexZ, G4ThreeVector& pos );


  int GetXIndex(G4double posX);
  int GetYIndex(G4double posY);
  int GetZIndex(G4double posZ);

protected:
  G4double   m_GateDiscretization;

private:
  GateDigi* m_outputDigi;

  GateDiscretizationMessenger *m_Messenger;

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








