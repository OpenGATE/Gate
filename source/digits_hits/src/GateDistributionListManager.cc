/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDistributionListManager.hh"
#include "GateDistributionListMessenger.hh"
#include "GateVDistribution.hh"

// Static pointer to the GateDistributionListManager singleton
GateDistributionListManager* GateDistributionListManager::theGateDistributionListManager=0;

/*    	This function allows to retrieve the current instance of the GateDistributionListManager singleton
      	If the GateDistributionListManager already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateDistributionListManager constructor
*/
GateDistributionListManager* GateDistributionListManager::GetInstance()
{
  Init();
  return theGateDistributionListManager;
}

void GateDistributionListManager::Init()
{
  if (!theGateDistributionListManager)
    theGateDistributionListManager = new GateDistributionListManager();
}

// Private constructor
GateDistributionListManager::GateDistributionListManager()
  : GateListManager( "distributions", "distributions", true, true )
{
  m_messenger = new GateDistributionListMessenger(this);
}



// Public destructor
GateDistributionListManager::~GateDistributionListManager()
{
  delete m_messenger;
}




// Registers a new object-Distribution in the Distribution list
void GateDistributionListManager::RegisterDistribution(GateVDistribution* newDistribution)
{
  theListOfNamedObject.push_back(newDistribution);
}
