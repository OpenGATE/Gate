/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateQuantumEfficiency.hh"

#include "GateQuantumEfficiencyMessenger.hh"
#include "GateTools.hh"

#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateVVolume.hh"

#include "GateMaps.hh"
#include "GateObjectStore.hh"

#include "G4UnitsTable.hh"


// Static pointer to the GateQuantumEfficiency singleton
GateQuantumEfficiency* GateQuantumEfficiency::theGateQuantumEfficiency=0;


/*    	This function allows to retrieve the current instance of the GateQuantumEfficiency singleton
      	If the GateQuantumEfficiency already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateQuantumEfficiency constructor
*/
GateQuantumEfficiency* GateQuantumEfficiency::GetInstance(GatePulseProcessorChain* itsChain,
							  const G4String& itsName)
{
  if (!theGateQuantumEfficiency)
    if (itsChain)
      theGateQuantumEfficiency = new GateQuantumEfficiency(itsChain, itsName);
  return theGateQuantumEfficiency;
}


// Private constructor
GateQuantumEfficiency::GateQuantumEfficiency(GatePulseProcessorChain* itsChain,
							  const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateQuantumEfficiencyMessenger(this);
  m_count = 0;
  m_testVolume = 0;
  m_nbTables = 0;
  m_uniqueQE = 1;
  m_nbFiles = 0;
  m_QECoef = 1;
}

// Public destructor
GateQuantumEfficiency::~GateQuantumEfficiency()
{
  delete m_messenger;
  delete [] m_table;
}



void GateQuantumEfficiency::CheckVolumeName(G4String val)
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
    G4cout << "Wrong Volume Name" << G4endl;
  }
}

void GateQuantumEfficiency::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if(!m_count)
    {
      if(!m_testVolume)
	{
	  G4cerr << 	G4endl << "[GateQuantumEfficiency::ProcessOnePulse]:" << G4endl
		 <<   "Sorry, but you don't have choosen any volume !" << G4endl;
	  G4Exception( "GateQuantumEfficiency::ProcessOnePulse", "ProcessOnePulse", FatalException, "You must choose a volume for crosstalk, e.g. crystal:\n\t/gate/digitizer/Singles/quantumEfficiency/chooseQEVolume VOLUME NAME\n or disable the quantumEfficiency using:\n\t/gate/digitizer/Singles/quantumEfficiency/disable\n");
	}
      //Create the table containing the quantum efficiencies
      CreateTable();
      m_count++;
    };
  m_depth = (size_t)(inputPulse->GetVolumeID().GetCreatorDepth(m_volumeName));
  std::vector<size_t> pulseLevels = m_levelFinder->FindInputPulseParams(&inputPulse->GetVolumeID(),
									  m_depth);
  m_volumeIDNo = pulseLevels[0];
  m_k = (pulseLevels.size() >= 1) ? pulseLevels[1] : 0;
  m_j = (pulseLevels.size() >= 2) ? pulseLevels[2] : 0;
  m_i = (pulseLevels.size() >= 3) ? pulseLevels[3] : 0;

  GatePulse* outputPulse = new GatePulse(*inputPulse);
  m_QECoef = m_table[m_k + m_j*m_level3No + m_i*m_level3No*m_level2No][m_volumeIDNo];
  outputPulse->SetEnergy(inputPulse->GetEnergy() * m_QECoef);
  outputPulseList.push_back(outputPulse);
}


void GateQuantumEfficiency::UseFile(G4String aFile)
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


void GateQuantumEfficiency::CreateTable()
{
  m_nbTables = m_level1No * m_level2No * m_level3No;
  G4int r, n;
  m_table = new G4double* [m_nbTables];
  std::ifstream in;
  std::ofstream out;

  if (nVerboseLevel > 1)
    G4cout << "Creation of a file called 'QETables.dat' which contains the quantum efficiencies tables" << G4endl;

  for (r = 0; r < m_nbTables; r++) {
    m_table[r] = new G4double [m_nbCrystals];
    if (m_nbFiles > 0) {
      size_t rmd=MonteCarloInt(0,m_nbFiles-1);
      in.open(m_file[rmd].c_str());
      for (n = 0; n < m_nbCrystals; n++){
	G4double rmd2 = MonteCarloG4double(-0.025,0.025);
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
	out << "#Table nb: " << r << G4endl;
	for (n = 0; n < (m_nbCrystals); n++)
	  out << m_table[r][n] << G4endl;
      }
  }
  if (out.is_open())
    out.close();
}


G4double GateQuantumEfficiency::MonteCarloEngine()
{
  G4double aleac;
  return (aleac = (((G4double) rand ()) / 2147483647.0));

}

size_t GateQuantumEfficiency::MonteCarloInt(size_t a,size_t b)
{
  size_t value;
  static size_t A,B;
  A = a;
  B = b+1;
  value =  ((size_t)(MonteCarloEngine()*(B-A))+A);


  return (value);

}

G4double GateQuantumEfficiency::MonteCarloG4double(G4double a,G4double b)
{
  G4double value;
  static G4double A,B;
  A = a;
  B = b;
  value =  ((G4double)(MonteCarloEngine()*(B-A))+A);


  return (value);

}

G4double GateQuantumEfficiency::GetMinQECoeff() {
  m_minQECoef=1.;
  for (G4int r = 0; r < m_nbTables; r++)
    for (G4int n = 0; n < m_nbCrystals; n++)
      m_minQECoef = (m_table[r][n] <= m_minQECoef) ?
	m_table[r][n] : m_minQECoef;

  return m_minQECoef;
}

void GateQuantumEfficiency::DescribeMyself(size_t indent)
{
  if (m_nbFiles > 0) {
    G4cout << GateTools::Indent(indent) << "Variable quantum efficiency based on the file(s): " << G4endl;
    std::vector<G4String>::iterator im;
    for (im=m_file.begin(); im!=m_file.end(); im++)
      G4cout << GateTools::Indent(indent+1) << *im << G4endl;
  }
  else
    G4cout << GateTools::Indent(indent) << "Fixed quantum efficiency equal to " << m_uniqueQE << G4endl;
}
