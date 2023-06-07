
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateTimeResolution

  	  ex- GateTemporalResolution by Martin.Rey@epfl.ch (July 2003)
    \brief  Digitizer Module modeling a Gaussian blurring (fwhm) on the time of the pulse.

   November 2022: Added to GND
*/

#include "GateTimeResolution.hh"
#include "GateTimeResolutionMessenger.hh"
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



GateTimeResolution::GateTimeResolution(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_fwhm(0),
   m_ctr(0),
   m_doi(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateTimeResolutionMessenger(this);
}


GateTimeResolution::~GateTimeResolution()
{
  delete m_Messenger;

}


void GateTimeResolution::Digitize()
{
	if (G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID() == 0)
		SetParameters();


	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();


	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;


  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  m_outputDigi = new GateDigi(*inputDigi);

		  G4double sigma =  m_fwhm / GateConstants::fwhm_to_sigma;
		  m_outputDigi->SetTime(G4RandGauss::shoot(inputDigi->GetTime(), sigma));

		  if (nVerboseLevel>1)
		  {
		    G4cout << "Digi real time: \n"
			   << G4BestUnit(inputDigi->GetTime(),"Time") << Gateendl
			   << "Digi new time: \n"
			   << G4BestUnit(m_outputDigi->GetTime(),"Time") << Gateendl
			   << "Difference (real - new time): \n"
			   << G4BestUnit(inputDigi->GetTime() - m_outputDigi->GetTime(),"Time")
			   << Gateendl << Gateendl ;

		  }

		  m_OutputDigiCollection->insert(m_outputDigi);


	  }
	}
  else
  {
	  if (nVerboseLevel>1)
	  	G4cout << "[GateTimeResolution::Digitize]: input digi collection is null -> nothing to do\n\n";
	    return;
  }
  StoreDigiCollection(m_OutputDigiCollection);

}

void GateTimeResolution::SetParameters()
{
	if(m_fwhm < 0 ) {
		G4cerr << 	Gateendl << "[GateTimeResolution::SetParameters]:\n"
      	   <<   "Sorry, but the negative resolution (" << GetFWHM() << ") is invalid\n";
    G4Exception( "GateTimeResolution::SetParameters", "SetParameters", FatalException,
			"You must choose a temporal resolution >= 0 .../timeResolution/fwhm TIME\n or disable the temporal resolution \n");
  }

	if(m_ctr < 0 ) {
		G4cerr << 	Gateendl << "[GateTimeResolution::SetParameters]:\n"
      	   <<   "Sorry, but the negative CTR resolution (" << GetFWHM() << ") is invalid\n";
    G4Exception( "GateTimeResolution::SetParameters", "SetParameters", FatalException,
			"You must choose a temporal resolution >= 0  .../timeResolution/ctr TIME\n or disable the temporal resolution\n");
  }
	if(m_fwhm != 0 && m_ctr != 0 ) {
		G4cerr << 	Gateendl << "[GateTimeResolution::SetParameters]:\n"
      	   <<   "Sorry, but you have to choose either FWHM or CTR for your time resolution \n";
    G4Exception( "GateTimeResolution::SetParameters", "SetParameters", FatalException,
			"Sorry, but you have to choose either FWHM or CTR for your time resolution\n");
  }

	if (m_ctr !=0 )
	{
		//from formula: CTR=sqrt (2*STR*STR+S*S),
		// CTR = coincidence time resolution
		// STR = single time resolution = m_fwhm
		// S = time spread due to geometry dimensions of the detector/DOI in this approximation
		// S = speed of light / DOI

		if (m_doi <= 0)
		{
			G4cerr << 	Gateendl << "[GateTimeResolution::SetParameters]:\n"
		      	   <<   "Sorry, but DOI either not set or not a positive non-null number \n";
		    G4Exception( "GateTimeResolution::SetParameters", "SetParameters", FatalException,
					"Sorry, but DOI either not set or not a positive non-null number\n");
		  }
		G4double S = m_doi*mm/c_light; // c_light is in mm/ns
		//G4cout<<"c_light= "<<c_light<<G4endl;
		//G4cout<<"S= "<<S*ns<<G4endl;
		//G4cout<<"m_ctr= "<<m_ctr*ns<<G4endl;
		m_fwhm = sqrt ( ((m_ctr*ns)*(m_ctr*ns) - S*S)/2);
		if (nVerboseLevel>1)
		{

			G4cout<<"[GateTimeResolution::SetParameters] The chosen Coincidence Time Resolution (CTR) is "<< m_ctr << " ns for a volume with a DOI size "<<m_doi<<" mm"<<G4endl;
			G4cout<<"[GateTimeResolution::SetParameters] The corresponding Single Time Resolution (STR) corresponding also to /fwhm is "<< m_fwhm <<" ns"<<G4endl;

		}
	}

}

void GateTimeResolution::DescribeMyself(size_t indent)
{
	  G4cout << GateTools::Indent(indent) << "Time resolution (fwhm): " << G4BestUnit(m_fwhm,"Time") << Gateendl;
}
