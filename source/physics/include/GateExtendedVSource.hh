/**
 *  @copyright Copyright 2016 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @file GateExtendedVSource.hh
 */
#ifndef GateExtendedVSource_hh
#define GateExtendedVSource_hh

#include "GateVSource.hh"
#include "GateGammaSourceModel.hh"
#include "GateGammaSourceModels.hh"
#include "GateExtendedVSourceMessenger.hh"

/**Author: Mateusz Bała
 * Email: bala.mateusz@gmail.com
 * About class: Extended version of GateVSource. It inherits all methods from GateVSource. It provides additional methods and solutions: support access to gammas source models.
 */
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
