/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEBACKTOBACK_HH
#define GATEBACKTOBACK_HH

#include "G4Event.hh"

#include "GateVSource.hh"

class GateBackToBack
{
public:
  GateBackToBack( GateVSource* );
  ~GateBackToBack();
	
  void Initialize();
  void GenerateVertex( G4Event*, G4bool);

private:
  GateVSource* m_source;
};

#endif
