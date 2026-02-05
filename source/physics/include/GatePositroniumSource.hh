/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
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
