
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateEnergyResolution

	(ex GateBlurring, crystal blurring, local energy blurring)

	 This module apples simulates Gaussian blurring of
	 the energy spectrum of a pulse after the readout module.


*/

#include "GateEnergyResolution.hh"
#include "GateEnergyResolutionMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateConstants.hh"



#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateEnergyResolution::GateEnergyResolution(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_reso(0),
   m_resoMin(0),
   m_resoMax(0),
   m_eref(0),
   m_slope(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);

	new G4UnitDefinition ( "1/electronvolt", "1/eV", "Energy Slope", 1/electronvolt );
	new G4UnitDefinition ( "1/kiloelectronvolt", "1/keV", "Energy Slope", 1/kiloelectronvolt );
	new G4UnitDefinition ( "1/megaelectronvolt", "1/MeV", "Energy Slope", 1/megaelectronvolt );
	new G4UnitDefinition ( "1/gigaelectronvolt", "1/GeV", "Energy Slope", 1/gigaelectronvolt );
	new G4UnitDefinition ( "1/joule", "1/J", "Energy Slope", 1/joule );

	m_Messenger = new GateEnergyResolutionMessenger(this);
 }


GateEnergyResolution::~GateEnergyResolution()
{
  delete m_Messenger;

}




void GateEnergyResolution::Digitize()
{

	if( m_resoMin!=0 && m_resoMax!=0 && m_reso!=0)
	{
		G4cout<<m_resoMin<<" "<< m_resoMax<<" "<<m_reso<<G4endl;
		GateError("***ERROR*** Energy Resolution is ambiguous: you can set /fwhm OR range for resolutions with /fwhmMin and /fwhmMax!");
	}



	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	G4double reso;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  if( m_resoMin!=0 && m_resoMax!=0)
			  reso = G4RandFlat::shoot(m_resoMin, m_resoMax);
		  else if (m_reso!=0)
			  reso=m_reso;



		  G4double energy= inputDigi->GetEnergy();
		  G4double sigma;
		  G4double resolution;

		  if (m_slope == 0 )
			  //Apply InverseSquareBlurringLaw
		  {
			  //G4cout<<"InverseSquareBlurringLaw"<<G4endl;
			  resolution = reso * sqrt(m_eref)/ sqrt(energy);

		  }
		  else
			  //Apply LinearBlurringLaw
			  resolution = m_slope * (energy - m_eref) + reso;

	      sigma =(resolution*energy)/GateConstants::fwhm_to_sigma;



		  G4double outEnergy=G4RandGauss::shoot(energy,sigma);

		  m_outputDigi = new GateDigi(*inputDigi);
		  m_outputDigi->SetEnergy(outEnergy);

		  if (nVerboseLevel>1)
		 	  G4cout << "[GateEnergyResolution::Digitize]: Created new digi from one with energy " << inputDigi->GetEnergy() << ".\n"
		 		 << "Resulting digi has energy: "<< m_outputDigi->GetEnergy() << Gateendl << Gateendl ;

		  m_OutputDigiCollection->insert(m_outputDigi);

	  }
    }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateEnergyResolution::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);


}



void GateEnergyResolution::DescribeMyself(size_t indent )
{
	  G4cout << GateTools::Indent(indent) << "Resolution of " << m_reso  << " for " <<  G4BestUnit(m_eref,"Energy") << Gateendl;
;
}
