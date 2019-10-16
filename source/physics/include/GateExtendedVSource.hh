/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateExtendedVSource_hh
#define GateExtendedVSource_hh

#include "GateVSource.hh"
#include "GateGammaSourceModel.hh"
#include "GateGammaSourceModels.hh"
#include "GateExtendedVSourceMessenger.hh"

/** Author: Mateusz Bała
 *  Email: bala.mateusz@gmail.com
 *  Organization: J-PET (http://koza.if.uj.edu.pl/pet/)
 *  About class: Extended version of GateVSource. It inherits all methods from GateVSource. It provides additional methods and solutions: support access to gammas source models.
 **/
class GateExtendedVSource : public GateVSource
{
public:
 /** Constructor
  * */
 GateExtendedVSource( G4String name );
 /** Destructor
  * */
 virtual ~GateExtendedVSource();

 /** If program first do not choose gamma source model this function call InitModel;
  * @param: event - event info
  * */
 virtual G4int GeneratePrimaries( G4Event* event ) override;

 void SetSeedForRandomGenerator( const unsigned int seed );

 void SetPromptGammaEnergy( const double energy );

 /** Function set linear polarization angle for particle
  * @param: angle - angle value (degree unit)
  * */
 void SetLinearPolarizationAngle( const double angle );

 /** Function set generation of unpolarized particles (what mean that particle has zero polarization vector {0,0,0})
  * @param: use_unpolarized - set true if you need unpolarized particles
  * */
 void SetUnpolarizedParticlesGenerating( const bool use_unpolarized );

protected:
 /** This function depends on user setting choose one correct model for simulation and associate with it pointer.
  * */
 bool InitModel();
 // This is pointer which will be associated with your model
 GateGammaSourceModel* ptrGammaSourceModel = nullptr;
 GateExtendedVSourceMessenger* pSourceMessenger = nullptr;
 unsigned int fSeedForRandomGenerator;
 double fPromptGammaEnergy = 0;// [keV]
 double fLinearPolarizationAngle = 0;// [keV]
 double fUseUnpolarizedParticles = false;
 

};


#endif
