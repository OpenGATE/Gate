/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFTableMgr.hh"
#include "GateARFTable.hh"
#include "GateARFTableMgrMessenger.hh"
#include "globals.hh"
#include "G4ios.hh"
#include <map>
#include <utility>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "GateARFSD.hh"


GateARFTableMgr::GateARFTableMgr( G4String aName, GateARFSD* theARFSD )
{
m_name = aName;
m_messenger = new GateARFTableMgrMessenger( aName , this );
m_EThreshHold = 0.;
m_EUpHold = 0.;
m_theARFSD = theARFSD; 
m_distance = 34.6*cm;
SaveARFTables = 0;
LoadARFTables = 0;
theFN = G4String("ARFTables.bin");
m_currentIndex = 0;
m_nbins = 100;
}

GateARFTableMgr::~GateARFTableMgr()
{
delete m_messenger;

}

G4double GateARFTableMgr::ScanTables(G4double x , G4double y , G4double theEnergy)
{
std::map<G4int,GateARFTable*>::iterator aIt;

for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{
  if ( (theEnergy - ( (*aIt).second )->GetElow() > 1.e-8 ) && (theEnergy - ( (*aIt).second )->GetEhigh() < 1.e-8 ) )
  {
    return ( ( (*aIt).second )->RetrieveProbability( x , y ) );
    break;
  }
}
return 0.;
}

void GateARFTableMgr::AddaTable( GateARFTable* aTable )
{
aTable->SetIndex( m_currentIndex );
m_theList.insert( std::make_pair(m_currentIndex, aTable) );
m_currentIndex++;
}

void GateARFTableMgr::ComputeARFTablesFromEW(G4String aS)
{
   m_filename = aS;

// read the spectrum data and construct as many ARF tables as energy intervals

  std::ifstream inFile( m_filename.c_str(), std::ios::in );

 if ( !inFile )
 {
	G4String AString = "Cannot open file "+m_filename;
  G4Exception( "GateARFTableMgr::ComputeARFTables", "ComputeARFTables", FatalException , AString);
	return;
 }

 char buffer [200];
  G4double Emin, Emax;
  G4int  NFiles ;
  G4String bname;
G4String basename = GetName()+"ARFTable_";

 while (!inFile.eof() )
 {

  inFile.getline(buffer,200);
  std::istringstream is(buffer);
    is.clear();
    is.str(buffer);
    G4String thestr = is.str();
    

    if ( thestr != "" && (thestr.find("#",0) != 0) && (thestr.find("!",0) != 0)  ) 
      {
       is >> Emin >> Emax>> bname >> NFiles ;
       
      GetARFSD()->AddNewEnergyWindow( bname, NFiles);
      std::ostringstream oss;
      oss << m_currentIndex;
      G4String thename = basename+oss.str();
      GateARFTable* theTable = new GateARFTable(thename);
      theTable->SetElow( Emin* keV  );
      theTable->SetEhigh( Emax* keV  );
      theTable->SetEnergyReso( m_EnReso);
      theTable->SetERef( m_ERef );
      theTable->SetDistanceFromSourceToDetector( m_distance );
      AddaTable( theTable );
      }
 }
 
m_theARFSD->computeTables();

    inFile.close();
}




G4int GateARFTableMgr::InitializeTables()
{

if ( LoadARFTables == 1 ) return 1;


std::map<G4int,GateARFTable*>::iterator aIt;

for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
( (*aIt).second )->Initialize(m_EThreshHold , m_EUpHold );

G4cout << "GateARFTableMgr::InitializeTables() : All ARF Tables are initialized " << G4endl;
return 0;
}
void GateARFTableMgr::SetNSimuPhotons(G4double* NPhotons)
{
std::map<G4int,GateARFTable*>::iterator aIt;
G4int i = 0;

for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{
 ( (*aIt).second )->SetNSimuPhotons( NPhotons[i]  );
i++;
}
}


void GateARFTableMgr::SetNBins(G4int N)
{
m_nbins = N;
}

void GateARFTableMgr::SetEReso(G4double aEnerReso)
{
std::map<G4int,GateARFTable*>::iterator aIt;

m_EnReso = aEnerReso;

for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{
 ( (*aIt).second )->SetEnergyReso( aEnerReso );
}
}


void GateARFTableMgr::SetERef(G4double aEnerReso)
{
std::map<G4int,GateARFTable*>::iterator aIt;

m_ERef = aEnerReso;

for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{
 ( (*aIt).second )->SetERef( aEnerReso );
}
}

void GateARFTableMgr::convertDRF2ARF()
{
std::map<G4int,GateARFTable*>::iterator aIt;
G4cout << " GateARFTableMgr::convertDRF2ARF()   CONVERTING DRF tables to ARF TABLES"<<G4endl;
for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{
//G4cout << " GateARFTableMgr::convertDRF2ARF()  flushing the raw DRF table to a binary file "<<G4endl;
 ( (*aIt).second )->convertDRF2ARF();
}
}


void GateARFTableMgr::FillDRFTable(G4int iT, G4double Edep, G4double Xprj, G4double Yprj)
{
std::map<G4int,GateARFTable*>::iterator aIt;
aIt = m_theList.find( iT );
if ( aIt != m_theList.end() ) ((*aIt).second )->FillDRFTable( Edep,  Xprj , Yprj );
else G4cout << " WARNING :: GateARFTableMgr::FillTable : Table # "<<iT<<" does not exist. Ignored "<<G4endl;


}

void GateARFTableMgr::ListTables()
{
std::map<G4int,GateARFTable*>::iterator aIt;
G4cout << " ----------------------------------------------------------------------------------------------"<<G4endl;
G4cout << " GateARFTableMgr::ListTables()  List of the ARF Tables for ARF Sensitive Detector " << GetName() << G4endl;

for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{ ( (*aIt).second )->Describe(); }
G4cout << " ----------------------------------------------------------------------------------------------"<<G4endl;
}

// save the ARF tables to a binary file

void GateARFTableMgr::SaveARFToBinaryFile()
{

ofstream dest;
dest.open ( theFN.c_str() );

std::map<G4int,GateARFTable*>::iterator aIt;

aIt = m_theList.begin();

    dest.seekp(0, std::ios::beg);

    G4double theSize = G4double(m_theList.size());

    dest.write( (const char*)(&theSize),sizeof(G4double) );

    theSize = G4double( ( (*aIt).second )->GetTotalNb() );

    long pos = dest.tellp();
    pos++;
    dest.write( (const char*)(&theSize),sizeof(G4double) );


    G4int theNb = 8 + G4int(theSize);

    size_t theBufferSize = theNb*sizeof(G4double);


for ( aIt = m_theList.begin() ; aIt != m_theList.end() ; aIt++ )
{
    G4double* theBuffer = new G4double[ theNb ];

    ( (*aIt).second )->GetARFAsBinaryBuffer( theBuffer );

//    size_t thePosition = theTableNb*theBufferSize;

    long pos = dest.tellp();
    pos++;

    dest.seekp( pos, std::ios::beg);

    if ( dest.bad() ) //G4Exception( "\nGateARFTableMgr::SaveARFToBinaryFile() :\n" "Could not locate the position where to write the ARFTable onto the disk (out of disk space?)!\n");
		{
			G4String msg = "Could not locate the position where to write the ARFTable onto the disk (out of disk space?)";
			G4Exception( "GateARFTableMgr::SaveARFToBinaryFile", "SaveARFToBinaryFile", FatalException, msg );
		}

    G4cout << " Writing ARF Table " <<( (*aIt).second )->GetName()<<" to file " <<theFN<<" at position " <<pos;

    dest.write((const char*)(theBuffer), theBufferSize );

    G4cout << "...[OK] : "<<theBufferSize<<" bytes"<<G4endl;

    if ( dest.bad() ) //G4Exception( "\nGateARFTableMgr::SaveARFToBinaryFile() :\n" "Could not write the ARFTable onto the disk (out of disk space?)!\n");
		{
			G4String msg = "Could not write the ARFTable onto the disk (out of disk space?)";
			G4Exception( "GateARFTableMgr::SaveARFToBinaryFile", "SaveARFToBinaryFile", FatalException, msg );
		}

    dest.flush();
    delete [] theBuffer;

G4cout << " All ARF tables have been saved to binary file\n " << theFN;
//G4Exception(".  Exiting.");
//exit(-1);
}

dest.close();

}


void GateARFTableMgr::LoadARFFromBinaryFile(G4String theFileName)
{

  LoadARFTables = 1;

  G4String basename = GetName()+"ARFTable_";

  ifstream dest;
  dest.open ( theFileName.c_str(),std::ios::binary );

    dest.seekg(0, std::ios::beg);

    G4double theNbOfTables, theSize;

    dest.read( (char*)(&theNbOfTables),sizeof(G4double) );
    long pos = dest.tellg();
    //pos++;

    G4cout << "Nb Of Tables "<<theNbOfTables<<G4endl;
    G4cout <<pos<<G4endl;

    dest.seekg( pos, std::ios::beg);
    dest.read( (char*)(&theSize),sizeof(G4double) );

    G4cout << "the size "<<theSize<<G4endl;
    G4cout <<pos<<G4endl;

    G4int theNb = 8 + int(theSize);
    size_t theBufferSize = theNb*sizeof(G4double);

    G4double* theBuffer = new G4double[ theNb ];

for ( size_t i =0  ; i < size_t(theNbOfTables) ; i++ )
{
    std::ostringstream oss;

    oss << m_currentIndex;

    G4String thename = basename+oss.str();

    long pos = dest.tellg();
    pos++;
    G4cout << " Reading ARF Table " <<thename<<" from file " <<theFileName<<" at position ";

    dest.seekg( pos, std::ios::beg);
    dest.read((char*)(theBuffer), theBufferSize );
    G4cout <<pos<< "...[OK] : "<<theBufferSize<<" bytes"<<G4endl;

    GateARFTable* theTable = new GateARFTable(thename);
    theTable->Initialize(theBuffer[4],theBuffer[5]);
    theTable->SetEnergyReso(theBuffer[2]);
    theTable->SetERef(theBuffer[3]);


    G4cout <<theBuffer[0]<<"  "<<theBuffer[1]<<"  "<<theBuffer[2]<<"  "<<theBuffer[3]<<"  "<<theBuffer[4]<<G4endl;
    G4cout << theBuffer[5]<<G4endl;

    theTable->FillTableFromBuffer( theBuffer );
    AddaTable( theTable );
   
}

dest.close();

    delete [] theBuffer;
    ListTables();

}

#endif

