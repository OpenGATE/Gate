
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateIntrinsicResolution

    \brief  Digitizer Module for simulating a special Gaussian blurring
	 ex GateBlurringWithIntrinsicResolution, GateLightYield, GateTransferEfficiency, GateQuantumEfficiency

    - GateIntrinsicResolution - by Martin.Rey@epfl.ch (mai 2003)

    - Digitizer Module for simulating a Gaussian blurring on the energy spectrum
    based on model (modeling with a parameterization of number of optical photons
    generated in a detector and the number of photoelectrons on a cathode :
    \f[R=\sqrt{\frac{1.1}{N_{ph}\cdot QE\cdot TE}\cdot 2.35^2+R_i^2}\f]
    where \f$N_{ph}=LY\cdot E_{inputHit}\f$.

   	Light Yield and Transfer Efficiency are also included in this class
   	since Gate9.3 (before they were separate classes)

    - Light Yield - the effect of the light yield
    - Transfer Efficiency - Allows to specify a transfer coefficients.
	- Quantum Efficiency - Allows to specify a quantum eff coefficients.

	4/12/23 - Added to GND by kochebina@cea.fr

      \sa GateVDigitizerModule
*/


#include "GateIntrinsicResolution.hh"
#include "GateIntrinsicResolutionMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateCrosstalk.hh"


#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateConstants.hh"
#include "GateObjectStore.hh"

GateIntrinsicResolution::GateIntrinsicResolution(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_resolution(-1),
   m_Eref(-1),
   m_LY(1),
   m_TE(1),
   m_QE(1),
  m_variance(0.1),
   m_edgesCrosstalkFraction(0),
   m_cornersCrosstalkFraction(0),
   isFirstEvent(true),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateIntrinsicResolutionMessenger(this);

	m_nbTables = 0;
	m_uniqueQE = 1;
	m_nbFiles = 0;

}


GateIntrinsicResolution::~GateIntrinsicResolution()
{
  delete m_Messenger;

}


void GateIntrinsicResolution::Digitize()
{

	if(isFirstEvent)
	{
	 if(m_resolution < 0 ) {
		 G4cerr << 	Gateendl << "[GateBlurringWithIntrinsicResolution::Digitize]:\n"
				 <<   "Sorry, but the resolution (" << m_resolution << ") for " << m_digitizer->GetSD()->GetName() << " is invalid\n";
		G4String msg = "You must set the energy of reference AND the resolution or disable the intrinsic resolution";
		G4Exception( "GateBlurringWithIntrinsicResolution::Digitize", "Digitize", FatalException, msg );
	 }
	 else if(m_Eref < 0) {
		 G4cerr <<   Gateendl << "[GateBlurringWithIntrinsicResolution::Digitize]:\n"
			       <<   "Sorry, but the energy of reference (" << G4BestUnit(m_Eref,"Energy") << ") for "
			       << m_digitizer->GetSD()->GetName() <<" is invalid\n";
		 G4String msg = "You must set the resolution AND the energy of reference or disable the intrinsic resolution";
		 G4Exception( "GateBlurringWithIntrinsicResolution::Digitize", "Digitize", FatalException, msg );
	 }

	 // For QE (from QE class)
     CheckVolumeName(m_digitizer->GetSD()->GetName());
     CreateTable();
     m_XtalkpCent = (1-(4*m_edgesCrosstalkFraction+4*m_cornersCrosstalkFraction));
   isFirstEvent=false;
   }

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;


	 if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateIntrinsicResolution::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
			    	for (size_t k=0; k<m_OutputDigiCollection->entries();k++)
			    		G4cout << *(*IDC)[k] << Gateendl;
			    		G4cout << Gateendl;
			    }




  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  // For QE from file
		  m_depth = (size_t)(inputDigi->GetVolumeID().GetCreatorDepth(m_digitizer->GetSD()->GetName()));
		  std::vector<size_t> pulseLevels = m_levelFinder->FindInputPulseParams(&inputDigi->GetVolumeID(),
		  									  m_depth);
		  m_volumeIDNo = pulseLevels[0];
		  m_k = (pulseLevels.size() >= 1) ? pulseLevels[1] : 0;
		  m_j = (pulseLevels.size() >= 2) ? pulseLevels[2] : 0;
		  m_i = (pulseLevels.size() >= 3) ? pulseLevels[3] : 0;

		  m_QE = m_table[m_k + m_j*m_level3No + m_i*m_level3No*m_level2No][m_volumeIDNo];
		  // end for QE


		  m_outputDigi = new GateDigi(*inputDigi);

		  G4double energy = inputDigi->GetEnergy();
		  G4double Nph = energy*m_LY*m_TE*m_QE*m_XtalkpCent;
		  G4double Ri = m_resolution * sqrt((m_Eref / energy));

		  G4double resol = sqrt((1+m_variance/Nph)*(GateConstants::fwhm_to_sigma*GateConstants::fwhm_to_sigma) + Ri*Ri);

		  m_outputDigi->SetEnergy(G4RandGauss::shoot(energy,(resol * energy)/GateConstants::fwhm_to_sigma));



		   if (nVerboseLevel>1)
		 	  G4cout << "Created new pulse for volume " << inputDigi->GetVolumeID() << ".\n"
		 		 << "Resulting pulse is: \n"
		 		 << *m_outputDigi << Gateendl << Gateendl ;

		  m_OutputDigiCollection->insert(m_outputDigi);


		if (nVerboseLevel==1) {
			G4cout << "[GateIntrinsicResolution::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateIntrinsicResolution::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


void GateIntrinsicResolution::UseFile(G4String aFile)
{
  std::ifstream in(aFile);
  G4double temp;
  G4int count = 0;
  while (1) {
    in >> temp;
    if (!in.good()) break;
    count++;
  }
  in.close();
  if (count==m_nbCrystals) {
    m_file.push_back(aFile);
    m_nbFiles++;
  }
}



void GateIntrinsicResolution::CreateTable()
{

  m_nbTables = m_level1No * m_level2No * m_level3No;
  G4int r, n;
  m_table = new G4double* [m_nbTables];
  std::ifstream in;
  std::ofstream out;

  if (nVerboseLevel > 1)
    G4cout << "Creation of a file called 'QETables.dat' which contains the quantum efficiencies tables\n";

  for (r = 0; r < m_nbTables; r++) {
    m_table[r] = new G4double [m_nbCrystals];
    if (m_nbFiles > 0) {
      size_t rmd = G4RandFlat::shootInt(m_nbFiles);
      in.open(m_file[rmd].c_str());
      for (n = 0; n < m_nbCrystals; n++){
	G4double rmd2 = G4RandFlat::shoot(-0.025, 0.025);
	in >> m_table[r][n];
	m_table[r][n]*=(rmd2+1);
      }
      in.close();
    }
    else
      for (n = 0; n < (m_nbCrystals); n++) {
	if (m_uniqueQE)
	  m_table[r][n] = m_uniqueQE;
	else
	  m_table[r][n] = 1;
      }
    if (nVerboseLevel > 1)
      {
	if (! out.is_open())
	  out.open("QETables.dat");
	out << "#Table nb: " << r << Gateendl;
	for (n = 0; n < (m_nbCrystals); n++)
	  out << m_table[r][n] << Gateendl;
      }
  }
  if (out.is_open())
    out.close();
}


void GateIntrinsicResolution::CheckVolumeName(G4String val)
{
  //Retrieve the inserter store to check if the volume name is valid
  GateObjectStore* anInserterStore = GateObjectStore::GetInstance();


  if (anInserterStore->FindCreator(val)) {
    m_volumeName = val;
    //Find the level params
    GateVVolume* anInserter = anInserterStore->FindCreator(m_volumeName);
    std::vector<size_t> levels;
    m_levelFinder = new GateLevelsFinder(anInserter, levels);
    m_nbCrystals = levels[0];
    m_level3No = (levels.size() >= 1) ? levels[1] : 1;
    m_level2No = (levels.size() >= 2) ? levels[2] : 1;
    m_level1No = (levels.size() >= 3) ? levels[3] : 1;

    m_testVolume = 1;

  }
  else {
    G4cout << "Wrong Volume Name\n";
  }
}

void GateIntrinsicResolution::DescribeMyself(size_t indent )
{
  ;
}
