/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDoIModels

  Digitizer module for simulating a DoI model
  The user can choose the axes for each tracked volume.

  It is an Adaptation of digitalization

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/

#include "GateDoIModels.hh"
#include "GateDoIModelsMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4UnitsTable.hh"
#include "GateTools.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateDoIModels::GateDoIModels(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_DoILaw(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	if (m_DoILaw == NULL)
	{
		G4String colName = digitizer->GetOutputName() ;
		collectionName.push_back(colName);
		m_Messenger = new GateDoIModelsMessenger(this);
		flgCorrectAxis=0;
	}
	else
		GateError("The effective energy is not set");
 }

void GateDoIModels::SetDoIAxis( G4ThreeVector val) {

    m_DoIaxis = val;

    G4ThreeVector xAxis(1.0,0,0.0);
    G4ThreeVector yAxis(0.0,1.0,0.0);
    G4ThreeVector zAxis(0.0,0.0,1.0);

    if(m_DoIaxis.isParallel(xAxis)||m_DoIaxis.isParallel(yAxis)||m_DoIaxis.isParallel(zAxis))
    {
        flgCorrectAxis=1;
    }
    else
    {
    	GateError("[GateDoIModels::GateDoIModels]:  one of the three axis must be selected  for DoI:X, Y or Z.");
    	GateError("[GateDoIModels::GateDoIModels]:  DoI model has not been applied.");
    }
}


GateDoIModels::~GateDoIModels()
{
  delete m_Messenger;

}


void GateDoIModels::Digitize()
{
	//G4cout<< "DoImodels = "<<flgCorrectAxis<<G4endl;

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();

	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;
	
	
    if (IDC)
	{
	   G4int n_digi = IDC->entries();

	   for (G4int i=0;i<n_digi;i++)
	   {
		   inputDigi=(*IDC)[i];

		   if (!inputDigi)
		   	{
		   	   if (nVerboseLevel>1)
		   	       G4cout << "[GateDoIModels::ProcessOnePulse]: input Digi was null -> nothing to do\n\n";
		   	   return;
		   	}

		   GateDigi* m_outputDigi = new GateDigi(*inputDigi);

		   if (flgCorrectAxis==1)
		   {
			   if (m_DoILaw == NULL)
			   {
				   GateError("The user did not set the effective energy law for the pulse");
			   }
			   m_DoILaw->ComputeDoI(m_outputDigi, m_DoIaxis);
		   }

		   m_OutputDigiCollection->insert(m_outputDigi);
		   if (nVerboseLevel>1)
		   	  G4cout << "Copied Digi to output:\n"
		   	         << *m_outputDigi << Gateendl << Gateendl ;
	   }

	   if (nVerboseLevel==1)
	   {
		   G4cout << "[GateDoIModels::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
		   for (size_t k=0; k<m_OutputDigiCollection->entries();k++)
			   G4cout << *(*IDC)[k] << Gateendl;
   		   G4cout << Gateendl;
	   }

	}

    StoreDigiCollection(m_OutputDigiCollection);
}



void GateDoIModels::DescribeMyself(size_t indent )
{
	G4cout << GateTools::Indent(indent) << "axis: (" << m_DoIaxis.getX()<<","<<m_DoIaxis.getY()<<","<<m_DoIaxis.getZ()<<Gateendl;
	  G4cout << GateTools::Indent(indent) << "law: " << m_DoILaw<<Gateendl;
}
