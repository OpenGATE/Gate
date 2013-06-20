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


namespace GateEnergyResponseFunctor
{

//-----------------------------------------------------------------------------
// Handling of the interpolation weight in primary: store the weights and

class InterpolationEnergyResponse
{
public:

  inline G4double operator() (double photonEnergy,
                              std::map<G4double,G4double> ResponseMap) {
      // Energy Response Detector (linear interpolation to obtain the right value from the list)
      std::map< G4double, G4double >::iterator iterResponseMap = ResponseMap.end();
      iterResponseMap =  ResponseMap.lower_bound( photonEnergy);
      if( iterResponseMap == ResponseMap.end()) {
         G4cout << " Photon Energy outside the Response Detector list" << G4endl;
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
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//template< class TInput1, class TInput2 = TInput1, class TOutput = TInput1 >
//class Attenuation
//{
//  public:
//  Attenuation() {}
//  ~Attenuation() {}
//  bool operator!=(const Attenuation &) const
//  {
//    return false;
//  }

//  bool operator==(const Attenuation & other) const
//  {
//    return !( *this != other );
//  }

//  inline TOutput operator()(const TInput1 A,const TInput2 B) const
//  {
//    //Calculating attenuation image (-log(primaryImage/flatFieldImage))
//    return (TOutput)(-log(A/B));
//  }
//};
//-----------------------------------------------------------------------------

}

#endif // GATEHYBRIDFORCEDDETECTIONACTORFUNCTORS_HH
