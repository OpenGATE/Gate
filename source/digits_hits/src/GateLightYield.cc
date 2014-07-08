/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLightYield.hh"
#include "GateLightYieldMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateObjectStore.hh"

// Static pointer to the GateLightYield singleton
GateLightYield* GateLightYield::theGateLightYield=0;



/*    	This function allows to retrieve the current instance of the GateLightYield singleton
      	If the GateLightYield already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateLightYield constructor
*/
GateLightYield* GateLightYield::GetInstance(GatePulseProcessorChain* itsChain,
					    const G4String& itsName)
{
  if (!theGateLightYield)
    if (itsChain)
      theGateLightYield = new GateLightYield(itsChain, itsName);
  return theGateLightYield;
}

// Private constructor
GateLightYield::GateLightYield(GatePulseProcessorChain* itsChain,
					    const G4String& itsName)
  : GateVPulseProcessor(itsChain, itsName)
{
  m_messenger = new GateLightYieldMessenger(this);
  m_lightOutput = 1.;
}

GateLightYield::~GateLightYield()
{
  delete m_messenger;
}

G4int GateLightYield::ChooseVolume(G4String val)
{
  //Retrieve the inserter store to check if the volume name is valid
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  if (m_store->FindCreator(val)!=0) {
    m_table[val] = 1.;
    return 1;
  }
  else {
    G4cout << "Wrong Volume Name" << G4endl;
    return 0;
  }
}

void GateLightYield::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  //Set the table iterator at the one which correspond to the layer volume name of the pulse
  im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
  GatePulse* outputPulse = new GatePulse(*inputPulse);
  if(im != m_table.end())
    {
      if((*im).second < 0 ) {
	G4cerr << 	G4endl << "[GateLightYield::ProcessOnePulse]:" << G4endl
	       <<   "Sorry, but the light output (" << (*im).second << ") for " << (*im).first << " is invalid" << G4endl;

	G4String msg = "You must set the light output:\n\t/gate/digitizer/Singles/lightYield/" + (*im).first + "/setLightOutput NBphotons/ENERGY\n or disable the light yield module using:\n\t/gate/digitizer/Singles/lightYield/disable";
	G4Exception( "GateLightYield::ProcessOnePulse", "ProcessOnePulse", FatalException, msg );
      }
      else {
	m_lightOutput = (*im).second;
	outputPulse->SetEnergy(inputPulse->GetEnergy() * m_lightOutput);
      }
    }
  outputPulseList.push_back(outputPulse);
}

G4double GateLightYield::GetMinLightOutput()
{
  im=m_table.begin();
  m_minLightOutput = (*im).second;
  m_minLightOutputName = (*im).first;
  for (im=m_table.begin(); im!=m_table.end(); im++)
    if ((*im).second <= m_minLightOutput)
      {
	m_minLightOutput = (*im).second;
	m_minLightOutputName = (*im).first;
      }
  return m_minLightOutput;
}

void GateLightYield::DescribeMyself(size_t indent)
{
  for (im=m_table.begin(); im!=m_table.end(); im++)
    G4cout << GateTools::Indent(indent) << (*im).first << " :\n"
	 << GateTools::Indent(indent+1) << "Light output: " << (*im).second * MeV << " photons/MeV\n" <<  G4endl;
}
