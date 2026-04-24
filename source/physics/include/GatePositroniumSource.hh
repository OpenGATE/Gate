/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About class: Geant4 primary event source for positronium annihilation; lazily initializes a GatePositroniumDecayModel from messenger-supplied parameters on the first event and delegates primary vertex generation to the model.
 **/

#ifndef GatePositroniumSource_hh
#define GatePositroniumSource_hh

#include <memory>

#include "GateVSource.hh"
#include "GatePositroniumSourceMessenger.hh"
#include "GateGammaEmissionModel.hh"

class GatePositroniumSource : public GateVSource
{
public:
  explicit GatePositroniumSource(const G4String& name);
  virtual ~GatePositroniumSource() = default;

  virtual G4int GeneratePrimaries( G4Event* event ) override;

 protected:
   void PrepareModel();

 protected:
  std::unique_ptr<GateGammaEmissionModel> pModel;
  std::unique_ptr<GatePositroniumSourceMessenger> pMessenger;
};

#endif
