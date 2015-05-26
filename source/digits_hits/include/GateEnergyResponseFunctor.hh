#ifndef GATEENERGYRESPONSEFUNCTOR_HH
#define GATEENERGYRESPONSEFUNCTOR_HH

// Geant4
#include <G4VEMDataSet.hh>
#include <G4EmCalculator.hh>
#include <G4VDataSetAlgorithm.hh>
#include <G4LivermoreComptonModel.hh>
#include <G4LogLogInterpolation.hh>
#include <G4CompositeEMDataSet.hh>
#include <G4CrossSectionHandler.hh>


//-----------------------------------------------------------------------------
// Handling of the interpolation weight in primary: store the weights and

class GateEnergyResponseFunctor
{
public:
inline G4double operator() (double photonEnergy) {
  if(!mResponseMap.size())
    return 1.;

  // Energy Response Detector (linear interpolation to obtain the right value from the list)
  std::map< G4double, G4double >::iterator iterResponseMap = mResponseMap.end();
  iterResponseMap =  mResponseMap.lower_bound( photonEnergy);
  if(iterResponseMap == mResponseMap.begin() || iterResponseMap == mResponseMap.end()) {
     G4cout << " Photon Energy outside the Response Detector list\n";
     exit(1);
  }
  double upperEn = iterResponseMap->first;
  double upperMu = iterResponseMap->second;
  iterResponseMap--;
  double lowerEn = iterResponseMap->first;
  double lowerMu = iterResponseMap->second;
  // Interpolation result value corresponding to the detector response
  double responseDetector = ((( upperMu - lowerMu)/( upperEn - lowerEn)) * ( photonEnergy - upperEn) + upperMu);
  return responseDetector;
}

void ReadResponseDetectorFile(std::string responseFileName) {
  if(responseFileName == "")
    return;

  G4double energy, response;
  std::ifstream inResponseFile;
  mResponseMap.clear();

  inResponseFile.open(responseFileName.c_str());
  if( !inResponseFile ) {
    // file couldn't be opened
    G4cout << "Error: file [" << responseFileName << "] could not be opened\n";
    exit( 1);
  }
  while ( !inResponseFile.eof( )) {
    inResponseFile >> energy >> response;
    energy = energy*MeV;
    mResponseMap[energy] = response;
  }
  inResponseFile.close( );
}

protected:
  std::map< G4double, G4double > mResponseMap;
};
//-----------------------------------------------------------------------------

#endif // GATEHYBRIDFORCEDDETECTIONACTORFUNCTORS_HH
