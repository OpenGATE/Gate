/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVisAttributesMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

#include "G4VVisManager.hh"

//------------------------------------------------------------------------------------------------------------------------------
#define N_COLORCODES 10
GateVisAttributesMessenger::GateColorPair GateVisAttributesMessenger::theColorTable[N_COLORCODES] = {
    GateColorPair ("white",      G4Colour(1.0, 1.0, 1.0)),
    GateColorPair ("gray",       G4Colour(0.5, 0.5, 0.5)),
    GateColorPair ("grey",       G4Colour(0.5, 0.5, 0.5)),
    GateColorPair ("black",      G4Colour(0.0, 0.0, 0.0)),
    GateColorPair ("red",        G4Colour(1.0, 0.0, 0.0)),
    GateColorPair ("green",      G4Colour(0.0, 1.0, 0.0)),
    GateColorPair ("blue",       G4Colour(0.0, 0.0, 1.0)),
    GateColorPair ("cyan",       G4Colour(0.0, 1.0, 1.0)),
    GateColorPair ("magenta",    G4Colour(1.0, 0.0, 1.0)),
    GateColorPair ("yellow",     G4Colour(1.0, 1.0, 0.0))
  };  
GateVisAttributesMessenger::GateColorMap GateVisAttributesMessenger::theColorMap = 
      GateColorMap(N_COLORCODES,theColorTable);
//------------------------------------------------------------------------------------------------------------------------------

      
//------------------------------------------------------------------------------------------------------------------------------
#define N_LINESTYLES 3
GateVisAttributesMessenger::GateLineStylePair GateVisAttributesMessenger::theLineStyleTable[N_LINESTYLES] = {
    GateLineStylePair ("unbroken",   G4VisAttributes::unbroken),
    GateLineStylePair ("dashed",     G4VisAttributes::dashed),
    GateLineStylePair ("dotted",     G4VisAttributes::dotted)
  };  
GateVisAttributesMessenger::GateLineStyleMap GateVisAttributesMessenger::theLineStyleMap = 
      GateLineStyleMap(N_LINESTYLES,theLineStyleTable);
//------------------------------------------------------------------------------------------------------------------------------

    
//------------------------------------------------------------------------------------------------------------------------------  
GateVisAttributesMessenger::GateVisAttributesMessenger(G4VisAttributes* itsVisAttributes,const G4String& itsName)
: GateMessenger(itsName),pVisAttributes(itsVisAttributes)
{ 
  G4String guidance = G4String("Control of visualisation attributes for a GATE volume");
  GetDirectory()->SetGuidance(guidance.c_str());

  G4String cmdName;

  cmdName = GetDirectoryName()+"setColor";
  pSetColorCmd = new G4UIcmdWithAString(cmdName,this);
  pSetColorCmd->SetGuidance("Selects the color for the volume.");
  pSetColorCmd->SetParameterName("choice",false);
  pSetColorCmd->SetCandidates(theColorMap.DumpMap(false,""," "));

  cmdName = GetDirectoryName()+"setVisible";
  pSetVisibilityCmd = new G4UIcmdWithABool(cmdName,this);
  pSetVisibilityCmd->SetGuidance("Shows or hides the volume.");
  pSetVisibilityCmd->SetParameterName("visibility-flag",true);
  pSetVisibilityCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"setDaughtersInvisible";
  pSetDaughtersInvisibleCmd = new G4UIcmdWithABool(cmdName,this);
  pSetDaughtersInvisibleCmd->SetGuidance("Shows or hides the volume's daughters.");
  pSetDaughtersInvisibleCmd->SetParameterName("invisibility-flag",true);
  pSetDaughtersInvisibleCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"setLineStyle";
  pSetLineStyleCmd = new G4UIcmdWithAString(cmdName,this);
  pSetLineStyleCmd->SetGuidance("Sets the volume's line-style.");
  pSetLineStyleCmd->SetParameterName("choice",false);
  pSetLineStyleCmd->SetCandidates(theLineStyleMap.DumpMap(false,""," "));

  cmdName = GetDirectoryName()+"setLineWidth";
  pSetLineWidthCmd = new G4UIcmdWithADouble(cmdName,this);
  pSetLineWidthCmd->SetGuidance("Sets the volume's line-width.");
  pSetLineWidthCmd->SetParameterName("Width",true);
  pSetLineWidthCmd->SetRange("Width>=0.");
  pSetLineWidthCmd->SetDefaultValue(1.);

  cmdName = GetDirectoryName()+"forceSolid";
  pForceSolidCmd = new G4UIcmdWithABool(cmdName,this);
  pForceSolidCmd->SetGuidance("force solid display for the volume.");
  pForceSolidCmd->SetParameterName("solid-flag",true);
  pForceSolidCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"forceWireframe";
  pForceWireframeCmd = new G4UIcmdWithABool(cmdName,this);
  pForceWireframeCmd->SetGuidance("force wireframe display for the volume.");
  pForceWireframeCmd->SetParameterName("wireframe-flag",true);
  pForceWireframeCmd->SetDefaultValue(true);

}
//------------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------------
GateVisAttributesMessenger::~GateVisAttributesMessenger()
{
  delete pSetColorCmd;
  delete pSetVisibilityCmd;
  delete pSetDaughtersInvisibleCmd;
  delete pSetLineStyleCmd;
  delete pSetLineWidthCmd;
  delete pForceSolidCmd;
  delete pForceWireframeCmd;
}
//------------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------------
void GateVisAttributesMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==pSetColorCmd )
    { SetColor(newValue); TelVisManagerToUpdate(); }   
  
  else if( command==pSetVisibilityCmd )
    { pVisAttributes->SetVisibility(pSetVisibilityCmd->GetNewBoolValue(newValue)); TelVisManagerToUpdate(); }   
  
  else if( command==pSetDaughtersInvisibleCmd )
    { pVisAttributes->SetDaughtersInvisible(pSetDaughtersInvisibleCmd->GetNewBoolValue(newValue)); TelVisManagerToUpdate();  }   
  
  else if( command==pSetLineStyleCmd )
    { SetLineStyle(newValue); TelVisManagerToUpdate(); }   
  
  else if( command==pSetLineWidthCmd )
    { pVisAttributes->SetLineWidth(pSetLineWidthCmd->GetNewDoubleValue(newValue)); TelVisManagerToUpdate();  }   
  
  else if( command==pForceSolidCmd )
    { pVisAttributes->SetForceSolid(pForceSolidCmd->GetNewBoolValue(newValue)); TelVisManagerToUpdate();  }   
  
  else if( command==pForceWireframeCmd )
    { pVisAttributes->SetForceWireframe(pForceWireframeCmd->GetNewBoolValue(newValue)); TelVisManagerToUpdate();  }   
  
  else
    GateMessenger::SetNewValue(command,newValue);
  
}
//------------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------------
void GateVisAttributesMessenger::SetColor(const G4String& colorString)
{
  GateColorMap::iterator colorMapIt = theColorMap.find(colorString);
  if ( colorMapIt == theColorMap.end()) {
    G4cout << "Color name '" << colorString << "' was not recognised --> ignored!\n";
    return;
    }

  pVisAttributes->SetColor(colorMapIt->second);
}
//------------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------------
void GateVisAttributesMessenger::SetLineStyle(const G4String& lineStyleName)
{
  GateLineStyleMap::iterator lineStyleMapIt = theLineStyleMap.find(lineStyleName);
  if ( lineStyleMapIt == theLineStyleMap.end()) {
    G4cout << "Line style name '" << lineStyleName << "' was not recognised --> ignored!\n";
    return;
    }
  pVisAttributes->SetLineStyle(lineStyleMapIt->second);
}
//------------------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------------------
void GateVisAttributesMessenger::TelVisManagerToUpdate()
{
  G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
  if(pVVisManager) pVVisManager->GeometryHasChanged();
}
//------------------------------------------------------------------------------------------------------------------------------
