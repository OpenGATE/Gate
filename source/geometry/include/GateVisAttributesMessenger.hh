/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVisAttributesMessenger_h
#define GateVisAttributesMessenger_h 1

#include "GateMessenger.hh"
#include "G4VisAttributes.hh"
#include "GateMaps.hh"

//-------------------------------------------------------------------------------------------------
class GateVisAttributesMessenger: public GateMessenger
{
public:
  GateVisAttributesMessenger(G4VisAttributes* itsVisAttributes,
			     const G4String& itsName);
  ~GateVisAttributesMessenger();
    
  void SetNewValue(G4UIcommand*, G4String);
  virtual inline G4VisAttributes* GetVisAttributes() { return pVisAttributes; }

private:
  virtual void SetColor(const G4String& colorName);
  virtual void SetLineStyle(const G4String& lineStyleName);
  virtual void TelVisManagerToUpdate();

private:
  typedef GateMap<G4String,G4Colour> GateColorMap ;
  typedef GateColorMap::MapPair GateColorPair ;

  static GateColorPair theColorTable[];
  static GateColorMap theColorMap;

  typedef GateMap<G4String,G4VisAttributes::LineStyle> GateLineStyleMap ;
  typedef GateLineStyleMap::MapPair GateLineStylePair ;

  static GateLineStylePair theLineStyleTable[];
  static GateLineStyleMap theLineStyleMap;

private:
  G4VisAttributes*       pVisAttributes;
  G4UIcmdWithAString*    pSetColorCmd;
  G4UIcmdWithABool* 	 pSetVisibilityCmd;
  G4UIcmdWithABool* 	 pSetDaughtersInvisibleCmd;
  G4UIcmdWithAString*    pSetLineStyleCmd;
  G4UIcmdWithADouble*    pSetLineWidthCmd;
  G4UIcmdWithABool*      pForceSolidCmd;
  G4UIcmdWithABool*      pForceWireframeCmd;
};
//-------------------------------------------------------------------------------------------------

#endif

