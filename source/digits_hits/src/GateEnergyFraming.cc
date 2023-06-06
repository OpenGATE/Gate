
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateEnergyFraming
  \brief  GateEnergyFraming applies an energy window selection
  ex-GateThresholder + ex-GateUpholder

  - GateEnergyFraming

  This digitizer module sets a simple
  uphold and threshold limits: any input digi whose energy is below or above
  the energy limit is copied into the output digi-list.
  On the contrary, any input digi whose energy is above or below this
  limit is discarded.

  Previous authors: Daniel.Strul@iphe.unil.ch, Steven.Staelens@rug.ac.be
  Added to GND in November 2022 by olga.kochebina@cea.fr
  // OK GND 2022
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateEnergyFraming.hh"
#include "GateEnergyFramingMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateEnergyFraming::GateEnergyFraming(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_EnergyFramingLaw(0),
   	m_min(0),
	m_max(std::numeric_limits<double>::max()),
	m_outputDigi(0),
	m_OutputDigiCollection(0),
	m_digitizer(digitizer)
   {

	G4String colName = digitizer->GetOutputName();
	collectionName.push_back(colName);
	m_Messenger = new GateEnergyFramingMessenger(this);
	std::vector<GateVolumeID> vID;

}




GateEnergyFraming::~GateEnergyFraming()
{
  delete m_Messenger;

}


void GateEnergyFraming::Digitize()
{
	//G4cout<< "EnergyFraming = "<<m_digitizer-> GetOutputName()<<G4endl;


	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();


	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi = new GateDigi();

	if (IDC)
     {
	  G4int n_digi = IDC->entries();
	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
		  {
		  	  G4double energy = 0;
			  inputDigi=(*IDC)[i];
			  GateDigi* outputDigi = new GateDigi(*inputDigi);

		      if( m_EnergyFramingLaw != 0 ){

		    	  energy = m_EnergyFramingLaw->ComputeEffectiveEnergy(*outputDigi);
		      }
		      else
		      {
		    	  energy = inputDigi->GetEnergy();
		      }
			  if ( energy >= m_min &&  energy <= m_max)
			    {
				  m_outputDigi = new GateDigi(*inputDigi);

				  m_OutputDigiCollection->insert(m_outputDigi);

			       if (nVerboseLevel>1)
			             	G4cout << "[GateEnergyFraming::Digitize] Copied digi to output:\n"
			             	       << *m_outputDigi << Gateendl << Gateendl ;
			    }
			  else
			  {
				  if (nVerboseLevel>1)
					  G4cout << "[GateEnergyFraming::Digitize]Ignored digi with energy above uphold:\n"
					  << *inputDigi << Gateendl << Gateendl ;
			  }

			  G4cout << inputDigi->GetEnergy() << G4endl;

		  }
    }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateEnergyFraming::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }

  StoreDigiCollection(m_OutputDigiCollection);
}



void GateEnergyFraming::DescribeMyself(size_t indent)
{
	  G4cout << GateTools::Indent(indent) << "EnergyFraming Min: " << G4BestUnit(m_min,"Energy") << Gateendl;
	  G4cout << GateTools::Indent(indent) << "EnergyFraming Max: " << G4BestUnit(m_max,"Energy") << Gateendl;

}
