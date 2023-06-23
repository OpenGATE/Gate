/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateClustering.cc for more details

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/

#ifndef GateClustering_h
#define GateClustering_h 1

#include <iostream>
#include <vector>
#include <set>
#include "G4ThreeVector.hh"

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateMaps.hh"
#include "GateClusteringMessenger.hh"
#include "GateSinglesDigitizer.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"


class GateClusteringMessenger;

class GateClustering : public GateVDigitizerModule
{
public:

  GateClustering(GateSinglesDigitizer *digitizer, G4String name);
  ~GateClustering();

  void Digitize() override;

  void SetClustering(G4double val)   { m_GateClustering = val; };

  void SetAcceptedDistance(G4double val) {  m_acceptedDistance = val;  };
  void SetRejectionFlag(G4bool flgval){m_flgMRejection=flgval;};

  void DescribeMyself(size_t );

protected:

  G4double   m_GateClustering;

  bool same_volumeID(const GateDigi* pulse1, const GateDigi* pulse2 );
  double getDistance(  G4ThreeVector pos1,G4ThreeVector pos2 );

  GateClusteringMessenger *m_Messenger;

private:
  GateDigi* m_outputDigi;



  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  G4double m_acceptedDistance;
  G4bool m_flgMRejection;



};

#endif








