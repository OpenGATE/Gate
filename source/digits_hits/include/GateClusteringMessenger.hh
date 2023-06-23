/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateDiscretizationMessenger

  \brief  Messenger for the GateDiscretization

  // OK GND 2022
  It is an Adaptation of digitalization
  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/

#ifndef GateClusteringMessenger_h
#define GateClusteringMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"

#include <vector>

class GateClustering;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateClustering;

class GateClusteringMessenger: public GateClockDependentMessenger
{
  public:

	GateClusteringMessenger(GateClustering*);
	~GateClusteringMessenger();

	void SetNewValue(G4UIcommand* aCommand, G4String aString);

private:

    G4UIcmdWithADoubleAndUnit*   pAcceptedDistCmd;
    G4UIcmdWithABool* pRejectionMultipleClustersCmd;
    G4UIcmdWithAString          *ClustCmd;
    GateClustering* m_GateClustering;
    G4String m_name;

};

#endif
