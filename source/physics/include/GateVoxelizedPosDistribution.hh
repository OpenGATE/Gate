/*
 * GateVoxelizedPosDistribution.hh
 *
 *  Created on: 20 nov. 2015
 *      Author: js244228
 */

#include "GateConfiguration.h"

#ifndef GATEVOXELIZEDPOSDISTRIBUTION_HH_
#define GATEVOXELIZEDPOSDISTRIBUTION_HH_

#include "globals.hh"
#include "GateSPSPosDistribution.hh"


class GateVoxelizedPosDistribution : public GateSPSPosDistribution
{
public:
  GateVoxelizedPosDistribution(G4String filename);
  ~GateVoxelizedPosDistribution();

  void SetPosition(G4ThreeVector pos) {mPosition = pos;};
  G4ThreeVector GenerateOne();

private:
  G4ThreeVector mPosition;       // position in world coordinates of pixel 0,0,0
  G4ThreeVector mResolution;     // resolution of data

  G4int m_nx, m_ny, m_nz;     // size of voxelized distribution
  G4double *mPosDistZCDF;   // Cumulative distribution function in Z
  G4double **mPosDistYCDF;  // CDF in Y
  G4double ***mPosDistXCDF;

};

#endif /* GATEVOXELIZEDPOSDISTRIBUTION_HH_ */
